# ref 1: https://github.com/Qybc/MedBLIP/blob/main/medblip/modeling_medblip_biomedlm.py
# ref 2: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py
# ref 3: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py
# ref 4: https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_t5_instruct.py
# ref 5: https://github.com/QwenLM/Qwen/blob/main/finetune.py#L172


import os 
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
import torch.distributed as dist
import pytorch_lightning as pl

from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from .eva_vit import create_eva_vit_g
from.openclip_vit import create_open_clip_vit_b, create_open_clip_vit_l
from utils.utils import scaling_lr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

import loralib as lora
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam



class Brain_BLIP(Blip2Base):
    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=128,
        patch_size=16,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        freeze_vit=True,
        lora_vit=False,
        num_query_token=32,
        cross_attention_freq=2,
        lm_tokenizer=None,
        lm_model=None,
        prompt="",
        max_txt_len=10000,
        apply_lemmatizer=False,
        embed_dim=256,
        precision='bf16'
    ):
        super().__init__()
        self.precision = precision
        ## initialize vision encoder
        if freeze_vit:
            assert lora_vit is False 
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                vit_model, img_size, patch_size, drop_path_rate, use_grad_checkpoint, use_lora=False
            )
            for name, param in self.visual_encoder.named_parameters():
                if 'blocks' in name:
                    param.requires_grad = False
                elif 'cls_' in name: 
                    param.requres_grad = False 
        elif lora_vit:
            assert freeze_vit is False 
            self.visual_encoder, self.ln_vision = self.init_vision_encoder(
                vit_model, img_size, patch_size, drop_path_rate, use_grad_checkpoint, use_lora=True
            )
            lora.mark_only_lora_as_trainable(self.visual_encoder, bias='all')
            for name, param in self.visual_encoder.named_parameters():
                if '3d' in name: 
                    param.requires_grad = True
        ## initialize qformer tokenizer
        self.tokenizer = self.init_tokenizer()

        ## initialize language encoder 
        self.lm_tokenizer = lm_tokenizer
        self.lm_model = lm_model
        for name, param in self.lm_model.named_parameters():
            param.requires_grad = False

        ## initialize Qformer 
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])
        
        
        ## initialize projection layers
        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.lm_proj = nn.Linear(self.Qformer.config.hidden_size, self.lm_model.config.hidden_size)

        self.proj = nn.Linear(self.Qformer.config.hidden_size, self.lm_model.config.hidden_size)
        self.temp = nn.Parameter(0.07 * torch.ones([]), requires_grad=False)
        self.max_txt_len = max_txt_len
        
    def init_vision_encoder(
        cls, 
        model_name, 
        img_size, 
        patch_size,
        drop_path_rate, 
        use_grad_checkpoint, 
        use_lora
    ):
        if model_name == "eva_clip_g":
            visual_encoder = create_eva_vit_g(
                    img_size,
                    patch_size, 
                    drop_path_rate, 
                    use_grad_checkpoint, 
                    use_lora=use_lora
                )
        
            ln_vision = nn.LayerNorm(visual_encoder.num_features)
        elif model_name == 'ViT-B-16': 
            visual_encoder = create_open_clip_vit_b(
                    img_size,
                    patch_size, 
                    drop_path_rate, 
                    use_grad_checkpoint, 
                    use_lora=use_lora
            )
            ln_vision = nn.LayerNorm(visual_encoder.num_features)
        return visual_encoder, ln_vision
    

    def concat_text_input_output(self, input_ids, input_atts, output_ids, output_atts):
        input_part_targets_len = []
        llm_tokens = {"input_ids": [], "attention_mask": []}
        for i in range(input_ids.size(0)):
            this_input_ones = input_atts[i].sum()
            input_part_targets_len.append(this_input_ones)
            llm_tokens['input_ids'].append(
                torch.cat([
                    input_ids[i][:this_input_ones],
                    output_ids[i][1:],
                    input_ids[i][this_input_ones:]
                ])
            )
            llm_tokens['attention_mask'].append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    output_atts[i][1:],
                    input_atts[i][this_input_ones:]
                ])
            )
        llm_tokens['input_ids'] = torch.stack(llm_tokens['input_ids'])
        llm_tokens['attention_mask'] = torch.stack(llm_tokens['attention_mask'])
        return llm_tokens, input_part_targets_len
    

    def lm_input_preprocess(self, 
                             input:list, 
                             text_type: str = 'instruction', # choices = ['instruction', 'question', 'answer']
                             system_message: str = "You are a helpful assistant.",
                             device=None): 
        """
        Tokenizing instruction is the same as query text from user in chatbot.
        Tokenizing answer is the same as generated text from assistans in chatbot.


        input: [[text 0], [text 1],... , [text B-1]]    # B: batch size
        output:
            - tokenized input text 
            - token id used for prediction

        => example text
        input_text0 = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{text0}\n"
        target_text0 ="<|im_start|><|im_end|>\n"
        """
        ## basic settings 
        IGNORE_TOKEN_ID = LabelSmoother.ignore_index
        roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}
        im_start = self.lm_tokenizer.im_start_id
        im_end = self.lm_tokenizer.im_end_id
        nl_tokens = self.lm_tokenizer('\n').input_ids
        _system = self.lm_tokenizer('system').input_ids + nl_tokens
        
        ## tokenize input text per batch 
        batch_inputs = [] 
        batch_targets = []
        for i in range(len(input)): 
            _input_tmp, _target_tmp = [], [] 
            system = [im_start] + _system + self.lm_tokenizer(system_message).input_ids + [im_end] + nl_tokens
            _input_tmp += system    # _input_tmp = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"]
            _target_tmp += [im_start] + [IGNORE_TOKEN_ID] * (len(system)-3) + [im_end] + nl_tokens     

            assert len(_input_tmp) == len(_target_tmp)
            #tokenizing inst or quest (In chatbot, it is query text from user)
            _input_text = self.lm_tokenizer(roles['user']).input_ids + nl_tokens + \
                self.lm_tokenizer(input[i]).input_ids + [im_end] + nl_tokens
            _input_tmp += _input_text    # input_tmp = ["<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{input_text}\n"]
            if text_type == 'instruction' or text_type == 'question': 
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_text)-3) + [im_end] + nl_tokens
            elif text_type == 'answer':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(self.lm_tokenizer(roles['user']).input_ids) + \
                    _input_text[len(self.lm_tokenizer(roles['assistant']).input_ids)+1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            _target_tmp += _target

            assert len(_input_tmp) == len(_target_tmp)
            _input_tmp += [self.lm_tokenizer.pad_token_id] * (self.max_txt_len - len(_input_tmp))
            _target_tmp += [IGNORE_TOKEN_ID] * (self.max_txt_len - len(_target_tmp))

            batch_inputs.append(_input_tmp[:self.max_txt_len])
            batch_targets.append(_target_tmp[:self.max_txt_len])

        batch_inputs = torch.tensor(batch_inputs, dtype=torch.int).to(device)
        batch_targets = torch.tensor(batch_targets, dtype=torch.int).to(device)
        batch_attention_masks = batch_inputs.ne(self.lm_tokenizer.pad_token_id)
        
        return batch_inputs, batch_targets, batch_attention_masks



    def forward(self, batch, global_rank=None): 
        """
        forward() method is the mixture of ref1 (https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_qformer.py) and 
        ref2 (https://github.com/salesforce/LAVIS/blob/main/lavis/models/blip2_models/blip2_opt.py)
        """
        """
        batch = {'image': torch.tensor, 'text': list, 'qna': list}
        """
        torch.cuda.empty_cache()
        image, text, inst, answer, label = batch['image'], batch['text'], batch['inst'], batch['answer'], batch['label']
        
        ## extract embeddings 
        # image embedding 
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # text embedding 
        self.tokenizer.truncation_side = 'right'
        text_tokens = self.tokenizer(
            [t + "\n" for t in text],   # add eos string
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids, 
            attention_mask=text_tokens.attention_mask, 
            return_dict=True,)
        
       

        # normalize extracted embeddings 
        image_feats = F.normalize(self.vision_proj(query_output.last_hidden_state), dim=-1)     # [batch_size, num_query_tokens, embed_dim]
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)     # [batch_size, embed_dim]


        ###============== Image-text Contrastive ===================###
        image_feats_all = concat_all_gather(image_feats)    # [batch_size*num_gpu, num_query_tokens, embed_dim]
        text_feat_all = concat_all_gather(text_feat)    # [batch_size*num_gpu, embed_dim]


        ### image and text  
        # image-text similarity: aggregate across all query tokens
        sim_q2t = torch.matmul(image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)).squeeze()     # [batch_size, batch_size*num_gpu, num_query_tokens]
        sim_i2t, _ = sim_q2t.max(-1)
        sim_i2t = sim_i2t / self.temp

        # text-image similarity: aggregate across all query tokens
        sim_t2q = torch.matmul(text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)).squeeze()
        sim_t2i, _ = sim_t2q.max(-1)
        sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

        rank = global_rank
        bs = image.size(0)
        targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
            image.device
        )

        loss_itc = (
                F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
            ) / 2
        

        ###============== Visual-aware instructions ===================###
        # query + instruction embedding
        self.tokenizer.truncation_side = 'left'
        text_Qformer = self.tokenizer(
            inst,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",).to(image.device)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(image.device)
        Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
        query_Qformer_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        
        inputs_llm = self.lm_proj(query_Qformer_output.last_hidden_state[:,:query_tokens.size(1),:])
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        # instruction embedding from frozen LLM
        self.lm_tokenizer.padding_side = "right"
        self.lm_tokenizer.truncation_side = 'left'
        text_input_tokens, text_input_targets, text_input_attention_masks = self.lm_input_preprocess(inst, text_type='instruction', device=image.device)

        # answer tokenizing with frozen LLM
        self.lm_tokenizer.truncation_side = 'right'
        text_output_tokens, text_output_targets, text_output_attention_masks = self.lm_input_preprocess(inst, text_type='answer', device=image.device)
        

        llm_tokens, input_part_targets_len = self.concat_text_input_output(
             text_input_tokens,
             text_input_attention_masks,
             text_output_tokens,
             text_output_attention_masks
         )


        # do not apply loss to the padding
        targets = llm_tokens['input_ids'].masked_fill(
            llm_tokens['input_ids'] == self.lm_tokenizer.pad_token_id, -100
        )

        # do not apply loss to the text input (i.e., instruction)
        for i, l in enumerate(input_part_targets_len):
            targets[i][:l] = -100

        # do not apply loss to the query tokens
        empty_targets = (
            torch.ones(atts_llm.size(), dtype=torch.long).to(image.device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.lm_model.get_input_embeddings()(llm_tokens['input_ids'])  
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, llm_tokens['attention_mask']], dim=1)

        outputs = self.lm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss_inst = outputs.loss

        loss=loss_itc+loss_inst
        
        loss_dict = {
            "loss_itc": loss_itc.mean().item(), 
            "loss_inst": loss_inst.mean().item(),
            "loss_total": loss.mean().item()
            }
        
        torch.cuda.empty_cache()
        return loss, loss_dict


    @torch.no_grad()
    def generate(
        self,
        batch,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
        device='cuda:0',
        ):
        # get image embedding
        image = batch["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # get image attended query embedding
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        inputs_llm = self.lm_proj(query_output.last_hidden_state)
        atts_llm = torch.ones(inputs_llm.size()[:-1], dtype=torch.long).to(image.device)

        # get text question tokens
        self.lm_tokenizer.padding_side = "left"
        text_input_tokens, _, text_input_attention_masks = self.lm_input_preprocess(batch['quest'], text_type='question', device=image.device)
        inputs_embeds = self.lm_model.get_input_embeddings()(text_input_tokens)
        
        # concat query and text question embedding
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, text_input_attention_masks], dim=1)   

        # generation with k-beam search
        generated_texts = self.batch_k_beam_search(model=self.lm_model, 
                                                   tokenizer=self.lm_tokenizer,
                                                   input_embeds=inputs_embeds,
                                                   attention_mask=attention_mask,
                                                   k=num_beams,
                                                   max_length=max_length,
                                                   min_length=min_length
                                                   )
    
        return generated_texts
    



    @torch.no_grad()
    def batch_k_beam_search(self, model, tokenizer, input_embeds, attention_mask, k=5, max_length=50, min_length=20):
        """
        Apply k-beam search sequentially to each sample in the batch.
        Args:
        - model: Pretrained language model.
        - tokenizer: Corresponding tokenizer.
        - input_texts (list of str): List of input texts.
        - k (int): Number of beams.
        - max_length (int): Maximum sequence length.
        - min_length (int): Minimum sequence length before allowing EOS token.
        
        Returns:
        - List of generated texts for each input text.
        """

        def _sequential_beam_search(model, tokenizer, input_embeds, attention_mask, k=5, max_length=50, min_length=20):
            """
            Perform k-beam search for a single input sequence represented by embeddings.
            Args:
            - model: Pretrained language model.
            - input_embeds (torch.Tensor): Input embeddings tensor.
            - attention_mask (torch.Tensor): Attention mask tensor.
            - k (int): Number of beams.
            - max_length (int): Maximum sequence length.
            - min_length (int): Minimum sequence length before allowing EOS token.
            
            Returns:
            - Tensor of token IDs for the best beam.
            """
            eos_token_id = model.config.eos_token_id
            device = input_embeds.device
            seq_len = input_embeds.size(1)
            embedding_dim = input_embeds.size(2)
            batch_size = input_embeds.size(0)

            # Initialize beams
            beams = input_embeds.repeat_interleave(k, dim=0)
            beams_attention_mask = attention_mask.repeat_interleave(k, dim=0)
            beam_scores = torch.zeros(batch_size * k, device=device)

            completed_beams = [[] for _ in range(batch_size)]
            completed_beam_scores = [torch.full((k,), -float("Inf"), device=device) for _ in range(batch_size)]

            for step in range(max_length - seq_len):
                outputs = model(inputs_embeds=beams, attention_mask=beams_attention_mask)
                logits = outputs.logits[:, -1, :]
                
                probs = torch.softmax(logits, dim=-1)
                
                # Prevent EOS selection if minimum length not achieved
                if step < min_length:
                    probs[:, eos_token_id] = 0
                    
                top_probs, top_indices = torch.topk(probs, k, dim=-1)
                
                # For the first step, expand each beam k times
                if step == 0:
                    beams = beams.unsqueeze(1).expand(-1, k, -1, -1).clone().reshape(batch_size * k * k, seq_len, embedding_dim)
                    beam_scores = beam_scores.unsqueeze(-1).expand(-1, k).clone().reshape(batch_size * k * k)
                    beams_attention_mask = beams_attention_mask.unsqueeze(1).expand(-1, k, -1).clone().reshape(batch_size * k * k, seq_len)
                        
                # Update beams, scores, and masks
                next_tokens = model.transformer.wte(top_indices)  # Convert token indices to embeddings
                beams = torch.cat([beams, next_tokens], dim=1)
                beam_scores += top_probs.log().view(-1)
                beams_attention_mask = torch.cat([beams_attention_mask, torch.ones((batch_size * k, 1), device=device)], dim=1)
                
                # Check and store completed beams
                for idx, beam in enumerate(beams):
                    if beam_scores[idx] != -float("Inf") and (top_indices.view(-1)[idx] == eos_token_id or step == max_length - seq_len - 1):
                        batch_idx = idx // k
                        completed_beams[batch_idx].append(beam)
                        completed_beam_scores[batch_idx][idx % k] = beam_scores[idx]
                        beam_scores[idx] = -float("Inf")
            # Handling cases where no beam is completed
            for i in range(batch_size):
                if not completed_beams[i]:  # If no completed beam, select the best available beam
                    beam_idx = beam_scores[i * k:(i + 1) * k].argmax()
                    completed_beams[i].append(beams[i * k + beam_idx])
                    completed_beam_scores[i][beam_idx] = beam_scores[i * k + beam_idx]


            # Select the best beams
            best_beams = [beams[i * k + completed_beam_scores[i].argmax()].unsqueeze(0) for i in range(batch_size)]
            best_beams_attention_mask = [beams_attention_mask[i * k + completed_beam_scores[i].argmax()].unsqueeze(0) for i in range(batch_size)]
            best_beams = torch.cat(best_beams, dim=0)
            best_beams_attention_mask = torch.cat(best_beams_attention_mask, dim=0)
            outputs = model(inputs_embeds=best_beams, attention_mask=best_beams_attention_mask)
            generated_token_id = outputs.logits[:, -1, :].argmax(axis=-1)
            return tokenizer.batch_decode(generated_token_id, skip_special_tokens=True)


        #generated_texts = [_sequential_beam_search(model, tokenizer, embed, mask, k, max_length, min_length) for embed, mask in zip(input_embeds, attention_mask)]
        generated_texts = _sequential_beam_search(model, tokenizer, input_embeds, attention_mask, k, max_length, min_length)
        return generated_texts




class Brain_BLIP_pl(pl.LightningModule): 
    def __init__(self, config: dict, lm_model, lm_tokenizer):
        """
        config: dictionary load from .yaml file
        """ 
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False

        # setting model
        self.model = Brain_BLIP(vit_model=self.hparams.model.image_encoder.vit_model,
                                img_size=self.hparams.model.image_encoder.img_size,
                                patch_size=self.hparams.model.image_encoder.patch_size,
                                drop_path_rate=self.hparams.model.image_encoder.drop_path_rate,
                                use_grad_checkpoint=self.hparams.model.image_encoder.use_grad_checkpoint,
                                freeze_vit=self.hparams.model.image_encoder.freeze_vit,
                                lora_vit=self.hparams.model.image_encoder.lora_vit,
                                num_query_token=self.hparams.model.image_encoder.num_query_token,
                                lm_tokenizer=lm_tokenizer,
                                lm_model=lm_model,
                                prompt=self.hparams.model.language_encoder.prompt,
                                max_txt_len=self.hparams.model.language_encoder.max_txt_len,
                                apply_lemmatizer=self.hparams.model.language_encoder.apply_lemmatizer,
                                embed_dim=self.hparams.model.language_encoder.embed_dim,
                                )
        
        # setting training hyperparameters 
        self.learning_rate = scaling_lr(batch_size=self.hparams.training_parameters.batch_size,
                                        accumulation_steps=self.hparams.training_parameters.accumulation_steps,
                                        base_lr=self.hparams.training_parameters.learning_rate)
        self.validation_step_outputs = None

    def training_step(self, batch, batch_idx): 
        loss, loss_dict = self.model(batch, int(self.global_rank))
        self.log_dict({
            "train_loss_itc": loss_dict['loss_itc'],
            "train_loss_inst": loss_dict['loss_inst'],
            "train_loss_total": loss_dict['loss_total']
        })
        torch.cuda.empty_cache()
        return loss
     

    def validation_step(self,batch, batch_idx): 
        _, loss_dict = self.model(batch, int(self.global_rank))
        self.log_dict({
            "valid_loss_itc": loss_dict['loss_itc'],
            "valid_loss_inst": loss_dict['loss_inst'],
            "valid_loss_total": loss_dict['loss_total']
        })
        """
        if batch_idx == 0: 
            sample_result_dict = {'input': batch,
                                  'answer': self.model.generate(batch, device=batch['image'].device)
                                  }
            self.validation_step_outputs = sample_result_dict
        self.validation_step_outputs = None
        """


    def test_step(self,batch, batch_idx): 
        _, loss_dict = self.model(batch, int(self.global_rank))
        self.log_dict({
            "test_loss_itc": loss_dict['loss_itc'],
            "test_loss_inst": loss_dict['loss_inst'],
            "test_loss_total": loss_dict['loss_total']
        })

    """
    def on_validation_epoch_end(self): 
        if self.validation_step_outputs is not None:
            input = self.validation_step_outputs['input']        
            inst = input['inst']
            quest = input['quest']
            answer= self.validation_step_outputs['answer']
            # make a result table and save the table
            columns = ['inst', 'quest', 'answer']
            data = list(zip(inst, quest, answer))
            self.logger.experiment.log_text(key="samples", columns=columns, data=data)
    """

    def configure_optimizers(self): 
        # setting optimizers
        if self.hparams.training_parameters.optimizer == "AdamW": 
            if self.hparams.pl_trainer.strategy == 'DeepSpeed_Zero3_offload':
                #optim = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, self.parameters()), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
                optim = DeepSpeedCPUAdam(self.parameters(), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
            else:
                #optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
                optim = torch.optim.AdamW(self.parameters(), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
            #optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
        else: 
            NotImplementedError("Only AdamW is implemented")
        
        # setting learning rate scheduler 
        if self.hparams.training_parameters.lr_scheduler == 'OneCycleLR': 
            sched = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=self.learning_rate, total_steps=self.trainer.estimated_stepping_batches)
            scheduler = {
                "scheduler": sched,
                "name": "lr_history",
                "interval": "step",
            }
            return [optim], [scheduler]
        else: 
            return optim




