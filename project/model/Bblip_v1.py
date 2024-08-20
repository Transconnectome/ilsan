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
        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)
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
                ])  # [input, output, padding]
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
        ##TODO 
        원래 BLIP 페이퍼에서는 2-stage training이 진행된다. 1) image-caption 간의 연결 능력을 학습시키는 과정, 2) 이미지만 주었을 때 생성시키는 걸 학습하는 과정. 
        이때 첫번째 과정의 핵심은 qformer가 a)이미지먹인 query와 텍스트를 매칭시키고(Image-text matching), b)이미지먹인 query로부터 텍스트 생성하기 (image-grounded text generation), c) 이미지먹인 query와 텍스트를 구분하기 (Image-text contrastive learning)이다. 
        두번째 과정은 이를 바탕으로 하여, qformer가 이미지 먹인 query만 주더라도 이미지에 맞는 text를 생성할 수 있도록 학습시키는 과정인 것이다.  

        Instruct BLIP은, 잘 학습된 BLIP을 활용하여 instruct라는 추가 정보를 주었을 때, 이 풍부한 정보를 활용하여 이미지에 맞는 텍스트를 더욱 잘 생성해내도록 만든 것이다. 

        그렇다면, 여기에서 임상가들의 소견과 라벨을 어떻게 다룰 것인가? 둘 중에 어떤 것을 first stage에 쓰고 second stage에 쓸 것인가? 혹은 둘 다를 쓸 것인가? 
        또한, BLIP에서의 second stage를 Instruct BLIP으로 대체할 것인가? 

        >>> Image-text matching: image + (report + answer) 
            Image-text contrastive learning: image + (report + answer) 
            Image-grounded text generation (visual-aware instruction learning ): image + instruct (report+question)  -> answer generation 


        
            

        - BLIP2와 Med BLIP의 차이
        원래 BLIP2의 프레임 워크를 현재 시나리오대로 풀어보자면, stage 1에서는 image-text(=report)만으로 qformer를 학습시키고, stage 2에서 Transformer 구조에 맞게 (i.e., autoregressive or seq2seq) qformer로부터 얻어낸 visual prompt로부터 text를 생성할 수 있는 능력(image-grounded text generation)을 학습시킨다. 
        즉, stage 1은 image와 text를 align 시키는 qformer를 학습시키고, stage 2에서는 qformer를 더욱 고도화하여 언어 모델이 image를 text prompt와 똑같이 visual prompt가 될 수 있도록 고도화시켜주는 것이다. 
        (추가로, stage 1과 stage 2에 쓰이는 image-text pair는 동일하다. 즉, stage 1에서 image-text align을 이후에, stage 2에서 QnA를 하는 게 아니라 stage 2에서도 동일하게 image-text align을 하는 것이다. 다만, 좀 더 고도화시키는 과정인 것이다. Instruct BLIP에서는 BLIP2 stage 1에서 학습된 모델을 활용해서, visual prompt에 instruction을 추가하여 훨씬 더 다양한 task들을 text generation task로 해결할 수 있게 고도화하는 작업인 것이다. 즉, 훨씬 더 발전된 형태의 stage 2 학습이라고 보면 된다.)
        하지만 Med BLIP에서는 한번의 learning stage만 거치게 되는데, 원래 BLIP2의 stage 1 학습에서 사용되는 세가지 loss인 1)image-text contrastive learning, 2)image-text matching, 3)image-grounded text generation 중에서 1)과 3)만을 활용한다. 
        이때, 3)의 loss 구하는 과정에서 BLIP2와 Med BLIP의 가장 큰 차이가 두드러진다. 

        BLIP2가 3)의 loss에서 input text를 causally genrate하는 것을 가르치는데 이때 image (정확히는 image 정보가 주입된 query embedding)을 key와 value로 주어서 seq2seq attention을 먹인다. 
        이는 multi-modal causal attention이라고도 한다. 이는 image는 self attend를 하고, text는 image를 cross attend 할 수 있는 반면에 image는 text를 cross attend할 수 없고, text는 causally self attend를 할 수 있는 형태이다. 
        이러한 attend 과정을 통한 학습이 진행되면서, text가 image 정보를 점점 활용할 수 있고 image는 text가 활용할 수 있는 형태로 점점 align이 되는 반면에, 그 반대로 text가 image에 align 되지는 않게 하면서 causally generation 할 수 있게 해주는 것이다. 
        즉, image가 text의 잠재 공간으로 점점 align이 되어가는 과정인 것이다.  
        이를 통해서 이미지의 정보를 활용해 text를 generation할 수 있는 능력을 학습시킨다. 
        
        Med BLIP의 경우, a)report+question으로 구성된 text, b)image (정확히는 image 정보가 주입된 query embedding), c)answer라는 세가지 input과 이들의 attention mask를 concatenate해서 언어 모델에 집어넣게 된다. 
        이때, 모델은 a)와 c)를 생성하도록 학습된다. 
        즉, a)와 b)와 c)가 합쳐진 하나의 embedding에서 self-attention이 일어나게 되며, a)와 c)의 label은 활용되고 b)의 label은 무시됨으로써 모델은 a)와 c)의 generation 생성 능력만을 평가하게 된다. 

        정리하자면, BLIP2의 stage 1과 Med BLIP은 image-grounded text generation loss의 구현 방식에서 아주 큰 차이가 있는데, 
        1) 언어 모델의 input에 image와 이를 설명하는 text만 넣었느냐 (=BLIP2; 우리 시나리오에서는 report text) 혹은 QnA의 text도 추가하였느냐 (=Med BLIP)
        2) image (정확히는 image 정보가 주입된 query embedding)와 text 정보들을 cross attention으로 합쳤느냐 (=BLIP2) 혹은 concatenate 했느냐 (=Med BLIP)   

        그런데, 자세히 살펴보면, Med BLIP의 image-grounded text generation 과정은 BLIP2의 stage 2에서 qformer를 autoregressive model (i.e., opt)에 적용하여 image-grounded text generation하는 것과 구조가 완전히 동일하다. 
        BLIP2의 stage 2에서 qformer를 autoregressive model에 합친 모델의 경우, autoregressive model의 input으로 image(정확히는 image 정보가 주입된 query embedding)와 text를 concatenate해서 언어 모델에 집어넣어준다. 
        그러면서 이렇게 합쳐진 하나의 embedding에서 self-attention이 일어나게 되며, image에 대한 label은 무시되고 text에 대한 label은 활용됨으로써 이미지 정보로부터 text를 생성할 수 있게 해준다. 
        즉, Med BLIP은, BLIP2의 stage 1에서의 image-text contrastive learning loss와 stage 2에서의 autoregressive model의 image-grounded text generation loss가 합쳐지게 된 형태이다. 

        최종적으로 요약하자면, 
        1) Med BLIP은, BLIP2의 stage 1에서의 image-text contrastive learning loss와 stage 2에서의 autoregressive model의 image-grounded text generation loss가 합쳐지게 된 형태로 loss function을 구성한다. (즉, Med BLIP은 BLIP2의 stage 1에서의 image-grounded text generation의 loss 형태가 아니라, stage 2에서의 image-grounded text generation의 loss 형태를 사용하는 것이다)
        2) Med BLIP은 BLIP2의 stage 2에서의 image-grounded text generation loss를 활용함에 있어서 원래의 text-image pair 데이터에서의 image로부터 text를 생성하게 하는 것이 아니라, 원래의 text-image pair 데이터에서 QnA text를 덧붙임으로써 text+image+question이 주어졌을 떄 answer를 생성할 수 있는 능력을 학습시킨다 (그리고 실제 학습할 때는 text, question, answer 모두를 잘 생성하도록 학습된다. text, image, question, answer가 주어지는 순서에 따른 ablation study를 한것이 Med BLIP 논문의 Table 4이다. )  

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


        ###============== Image-text Contrastive (Uni-modal Self-Attention) ===================###
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
        

        ###============== Image-text matching (Image and text multi-modal Self-Attention)===================###
        text_input_ids_world = concat_all_gather(text_tokens.input_ids)
        text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
        image_embeds_world = all_gather_with_grad(image_embeds)
        with torch.no_grad():
            sim_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)
            sim_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(-10000)   
            weights_t2i = F.softmax(sim_t2i, dim=1)
            weights_i2t = F.softmax(sim_i2t, dim=1)
        
        # select a negative image for each text
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds_world[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image
        text_ids_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_ids_neg.append(text_input_ids_world[neg_idx])
            text_atts_neg.append(text_attention_mask_world[neg_idx])
        text_ids_neg = torch.stack(text_ids_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)
        text_ids_all = torch.cat(
            [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
        )  # pos, pos, neg
        text_atts_all = torch.cat(
            [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
            dim=0,
        )

        # get query token and attention mask
        query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
        query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
            image.device
        )
        attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

        # get image embedding and attention mask
        image_embeds_all = torch.cat(
            [image_embeds, image_embeds_neg, image_embeds], dim=0
        )  # pos, neg, pos
        image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
            image.device
        )

        # forward Qformer
        output_itm = self.Qformer.bert(
            text_ids_all,
            query_embeds=query_tokens_itm,
            attention_mask=attention_mask_all,
            encoder_hidden_states=image_embeds_all,
            encoder_attention_mask=image_atts_all,
            return_dict=True,
        )

        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
        vl_output = self.itm_head(vl_embeddings)
        logits = vl_output.mean(dim=1)

        itm_labels = torch.cat(
            [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
            dim=0,
        ).to(image.device)
        loss_itm = F.cross_entropy(logits, itm_labels)
        

        ###============== Image-grounded text generation (Visual-aware instruction learning; Multi-modal Causal Self-Attention) ===================###
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


        ###============== summarize losses ===================###
        # sum all kinds of loss
        loss=loss_itc+loss_itm+loss_inst
        
        loss_dict = {
            "loss_itc": loss_itc.mean().item(), 
            "loss_itm": loss_itc.mean().item(), 
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
        num_beams=5,
        max_length=30,
        min_length=5,
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
        text_input_tokens, _, text_input_attention_masks = self.lm_input_preprocess(batch['inst'], text_type='instruction', device=image.device)
        inputs_embeds = self.lm_model.get_input_embeddings()(text_input_tokens)
        
        # concat query and text question embedding
        inputs_embeds = torch.cat([inputs_llm, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llm, text_input_attention_masks], dim=1)   
        
        # generation with k-beam search
        """
        generated_sequences = self.generate_decode_with_k_beam_search(model=self.lm_model, 
                                                                    tokenizer=self.lm_tokenizer,
                                                                    input_embeds=inputs_embeds,
                                                                    attention_mask=attention_mask,
                                                                    k=num_beams,
                                                                    max_length=max_length,
                                                                    min_length=min_length
                                                                    )
        """
        generated_sequences = self.generate_decode(model=self.lm_model, 
                                                     tokenizer=self.lm_tokenizer,
                                                     input_embeds=inputs_embeds,
                                                     attention_mask=attention_mask,
                                                     max_length=max_length,
                                                     min_length=min_length
                                                     )
        
        print(generated_sequences)
        return generated_sequences
    
    
    @torch.no_grad()
    def generate_decode(self, model, tokenizer, input_embeds, attention_mask, max_length=10, min_length=2):
        eos_token_id = model.config.eos_token_id
        device = input_embeds.device
        seq_len = input_embeds.size(1)
        embedding_dim = input_embeds.size(2)
        batch_size = input_embeds.size(0)

        generated_token_id = []
        finished = torch.zeros(batch_size, dtype=torch.bool)
        current_length = torch.zeros(batch_size, dtype=torch.long)
        for _ in range(max_length):
            outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            predictions = outputs.logits
            predictions = torch.softmax(predictions, dim=-1)
            predicted_ids = torch.argmax(predictions[:, -1, :], dim=-1).unsqueeze(-1)
            generated_token_id.append(predicted_ids)

            # 생성된 시퀀스의 길이 업데이트
            current_length += 1
            
            # EOS 토큰 생성 확인 및 최소 길이 조건 검사
            is_eos_generated = (predicted_ids.squeeze(-1) == eos_token_id)
            can_finish = (current_length >= min_length)
            finished |= is_eos_generated & can_finish
            if finished.all():
                break

            next_embedding = self.lm_model.get_input_embeddings()(predicted_ids)
            next_attention_mask = torch.ones((batch_size, 1), device=attention_mask.device)
            input_embeds = torch.cat([input_embeds, next_embedding], dim=1)
            attention_mask = torch.cat([attention_mask, next_attention_mask], dim=1)

        generated_token_id = torch.cat(generated_token_id, dim=-1)
        return tokenizer.batch_decode(generated_token_id, skip_special_tokens=True)
        




    @torch.no_grad()
    def generate_decode_with_k_beam_search(self, model, tokenizer, input_embeds, attention_mask, k=5, max_length=10, min_length=2):
        eos_token_id = model.config.eos_token_id
        pad_token_id = tokenizer.pad_token_id
        device = input_embeds.device
        batch_size = input_embeds.size(0)

        # 입력 임베딩과 어텐션 마스크 초기화
        input_embeds = input_embeds.unsqueeze(1).expand(-1, k, -1, -1).contiguous().view(batch_size * k, -1, input_embeds.size(-1))
        attention_mask = attention_mask.unsqueeze(1).expand(-1, k, -1).contiguous().view(batch_size * k, -1)

        beam_scores = torch.zeros(batch_size * k, device=device)
        beam_tokens = torch.full((batch_size * k, max_length), pad_token_id, dtype=torch.long, device=device)
        beam_lengths = torch.zeros(batch_size * k, dtype=torch.long, device=device)
        beam_eos_mask = torch.zeros(batch_size * k, dtype=torch.bool, device=device)

        for step in range(max_length):
            if step > 0:
                input_embeds = torch.cat([input_embeds, model.get_input_embeddings()(beam_tokens[:, step-1:step])], dim=1)
                attention_mask = torch.cat([attention_mask, torch.ones((batch_size * k, 1), device=device)], dim=1)

            outputs = model(inputs_embeds=input_embeds, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]
            logits = torch.softmax(logits, dim=-1)
            topk_scores, topk_tokens = torch.topk(logits, k, dim=-1)

            # 빔 점수 및 토큰 업데이트
            if step == 0:
                beam_scores = topk_scores.view(batch_size * k, k).contiguous().view(-1)
            else:
                beam_scores += topk_scores.view(batch_size * k, k).max(dim=1)[0]  # 최대 점수만 더함

            # 각 빔의 현재 길이 업데이트
            beam_lengths += 1
            # EOS 토큰이 나타났는지 체크하고, 최소 길이에 도달하지 않았으면 무시
            is_eos = (topk_tokens == eos_token_id)
            print(is_eos)
            beam_eos_mask |= is_eos.view(-1)
            too_short = (beam_lengths < min_length)
            beam_tokens[:, step] = torch.where(too_short, topk_tokens.view(-1), eos_token_id if is_eos else topk_tokens.view(-1))

        # 최종 시퀀스 선택 및 디코딩
        best_indices = beam_scores.view(batch_size, k).argmax(dim=1)
        best_sequences = beam_tokens.view(batch_size, k, -1)[torch.arange(batch_size), best_indices]
        return [tokenizer.decode(seq, skip_special_tokens=True) for seq in best_sequences]










class Brain_BLIP_pl(pl.LightningModule): 
    def __init__(self, config: dict, img_size=None, lm_model=None, lm_tokenizer=None):
        """
        config: dictionary load from .yaml file
        """ 
        super().__init__()
        self.save_hyperparameters(config)
        self.automatic_optimization = False

        # setting model
        self.model = Brain_BLIP(vit_model=self.hparams.model.image_encoder.vit_model,
                                img_size=img_size,
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
            "train/loss_itc": loss_dict['loss_itc'],
            "train/loss_itm": loss_dict['loss_itm'],
            "train/loss_inst": loss_dict['loss_inst'],
            "train/loss_total": loss_dict['loss_total']
        })
        if batch_idx == 0: 
            sample_result_dict = {'input': batch,
                                  'answer': self.model.generate(batch, device=batch['image'].device)
                                  }
        torch.cuda.empty_cache()
        return loss
     

    def validation_step(self,batch, batch_idx): 
        _, loss_dict = self.model(batch, int(self.global_rank))
        self.log_dict({
            "valid/loss_itc": loss_dict['loss_itc'],
            "valid/loss_itm": loss_dict['loss_itm'],
            "valid/loss_inst": loss_dict['loss_inst'],
            "valid/loss_total": loss_dict['loss_total']
        })
        
        if batch_idx == 0: 
            sample_result_dict = {'input': batch,
            #                      'answer': self.model.generate(batch, device=batch['image'].device)
                                  }
            self.validation_step_outputs = sample_result_dict
        self.validation_step_outputs = None
        


    def test_step(self,batch, batch_idx): 
        _, loss_dict = self.model(batch, int(self.global_rank))
        self.log_dict({
            "test/loss_itc": loss_dict['loss_itc'],
            "test/loss_itm": loss_dict['loss_itm'],
            "test/loss_inst": loss_dict['loss_inst'],
            "vest/loss_total": loss_dict['loss_total']
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




