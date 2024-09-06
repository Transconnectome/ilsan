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

from lavis.models import load_model
from timm.models.layers import drop_path, to_3tuple, trunc_normal_
from lavis.models.blip2_models.blip2 import Blip2Base
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from .eva_vit import create_eva_vit_g, PatchEmbed

from utils.utils import scaling_lr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

import loralib as lora


from sklearn.metrics import accuracy_score, roc_auc_score, r2_score




class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, dtype=torch.float32):
        super().__init__()
        img_size = to_3tuple(img_size)
        patch_size = to_3tuple(patch_size)
        num_patches = (img_size[2] // patch_size[2]) * (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, dtype=dtype)

        self.positional_embeddings = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        trunc_normal_(self.positional_embeddings, std=.02)

    def forward(self, x, interpolate_pos_encoding=False):
        B, C, D, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert D == self.img_size[0] and H == self.img_size[1] and W == self.img_size[2], \
            f"Input image size ({D}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = x + self.positional_embeddings
        return x




class Brain_BLIP(Blip2Base):
    def __init__(
        self,
        model_arch="blip2_t5",
        model_type="pretrain_flant5xl",
        img_size=128,
        lora_vit=False, 
        lora_llm=False,
        max_txt_len=None,
    ):
        super().__init__()
        ### setting model
        self.model = load_model(name=model_arch , model_type=model_type, is_eval=True, device='cpu')

        ### replace original 2D model's patch embedding layer and positional embedding with 3D patch embedding layer and positional embedding
        # make new 3D patch embedding layer and positional embedding
        patch_embed_3d = PatchEmbed(
            img_size=img_size, 
            #patch_size=self.model.visual_encoder.patch_embed.proj.kernel_size[0], 
            patch_size=18,
            in_chans=1, 
            embed_dim=int(self.model.visual_encoder.patch_embed.proj.out_channels))
        num_patches = patch_embed_3d.num_patches
        pos_embed_3d = nn.Parameter(torch.zeros(1, num_patches + 1, int(self.model.visual_encoder.patch_embed.proj.out_channels)))
        trunc_normal_(pos_embed_3d, std=.02)
        #pass through original 2D model's patch embedding layer and positional embedding 
        setattr(self.model.visual_encoder, "patch_embed", patch_embed_3d)
        setattr(self.model.visual_encoder,"pos_embed", pos_embed_3d)

        ### setting language encoder hyperparameters 
        self.max_txt_len = max_txt_len

        # for hugging face tokenizer
        os.environ['TOKENIZERS_PARALLELISM'] = "false"

        ### freeze parameters 
        # freeze every parameters except for patch embedding and positional embedding layer 
        for name, param in self.model.visual_encoder.named_parameters():
            if 'blocks' in name:
                param.requires_grad = False
            if 'cls_' in name: 
                param.requres_grad = False 
            if 'pos_embed' in name: 
                param.requires_grad = True 
            if 'patch_embed_' in name: 
                param.requires_grad = True
        # freeze Qformer
        for name, param in self.model.named_parameters():
            if 'Qformer' in name:
                param.requires_grad = False
            if 't5_proj' in name:
                param.requires_grad = False
        # freeze query token 
        for name, param in self.model.named_parameters():
            if 'query_tokens' in name:
                param.requires_grad = False
        for name, param in self.model.named_parameters():
            if 't5_model' in name:
                param.requires_grad = False
        for name, param in self.model.t5_model.named_parameters():
            if "lm_head" in name:
                param.requires_grad = False
 


    def forward(self, batch, global_rank=None): 
        #torch.cuda.empty_cache()
        #change the key name
        #batch['text_input'], batch['text_output'] = batch['inst'], batch['answer']
        #del batch['inst']
        #del batch['answer']
        loss_dict = self.model.forward(batch)
        pred = self.generate(batch)
        #pred = pred.detach().cpu().tolist()

        ### for sex classification
        #pred = [0 if sex == 'male' else 1 for sex in pred]
        ### for age classification    
        
        torch.cuda.empty_cache()
        return loss_dict['loss'], loss_dict, pred


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
        batch['prompt'] = batch['text_input']
        #del batch['inst']
        output_text = self.model.generate(batch)
        #print(f"GT: {batch['answer']}\nPRED:{output_text}")
        return output_text



class Brain_BLIP_pl(pl.LightningModule): 
    def __init__(self, config: dict, img_size=None):
        """
        config: dictionary load from .yaml file
        """ 
        super().__init__()
        self.save_hyperparameters(config)
        #self.automatic_optimization = False
        self.model = Brain_BLIP(model_arch=self.hparams.model.architecture,
                                model_type=self.hparams.model.type,
                                img_size=img_size,
                                lora_vit=self.hparams.model.image_encoder.lora_vit,
                                lora_llm=self.hparams.model.language_encoder.lora_llm,
                                max_txt_len=self.hparams.model.language_encoder.max_txt_len,
                                )
        

        
        # setting training hyperparameters 
        #self.learning_rate = scaling_lr(batch_size=self.hparams.training_parameters.batch_size,
        #                                accumulation_steps=self.hparams.training_parameters.accumulation_steps,
        #                                base_lr=self.hparams.training_parameters.learning_rate)
        self.learning_rate = self.hparams.training_parameters.learning_rate
        self.stage2_start_epoch = self.hparams.pl_trainer.stage2_start_epoch
        self.stage3_start_epoch = self.hparams.pl_trainer.stage3_start_epoch
        self.validation_step_outputs = None
        self.automatic_optimization = False


    def summarize_model_performance(self, target, pred, stage): 
        def _one_hot_encoder(text, stage):
            """
            text	label
            This subject does not have any brain lesion.	0
            This subject has an Acute brain lesion.	1
            This subject has an Acute brain lesion characterized by hemorrhage.	2
            This subject has an Acute brain lesion characterized by ICH.	3
            This subject has an Acute brain lesion characterized by infarction.	4
            This subject has an Recent brain lesion.	5
            This subject has an Recent brain lesion characterized by hemorrhage.	6
            This subject has an Recent brain lesion characterized by ICH.	7
            This subject has an Recent brain lesion characterized by infarction.	8
            This subject has an Chronic brain lesion.	9
            This subject has an Chronic brain lesion characterized by hemorrhage.	10
            This subject has an Chronic brain lesion characterized by ICH.	11
            This subject has an Chronic brain lesion characterized by infarction.	12
            This subject has an Old brain lesion.	13
            This subject has an Old brain lesion characterized by hemorrhage.	14
            This subject has an Old brain lesion characterized by ICH.	15
            This subject has an Old brain lesion characterized by infarction.	16

            """
            """
            if "This subject does not have any brain lesion." in text:
                value = 0 
            elif "This subject has an Acute brain lesion." in text:
                value = 1
            elif "This subject has an Acute brain lesion characterized by hemorrhage." in text:
                value = 2 
            elif "This subject has an Acute brain lesion characterized by ICH." in text:
                value = 3 
            elif "This subject has an Acute brain lesion characterized by infarction." in text:
                value = 4 
            elif "This subject has an Recent brain lesion." in text:
                value = 1
            elif "This subject has an Recent brain lesion characterized by hemorrhage." in text:
                value = 2 
            elif "This subject has an Recent brain lesion characterized by ICH." in text:
                value = 3 
            elif "This subject has an Recent brain lesion characterized by infarction." in text:
                value = 4 
            elif "This subject has an Chronic brain lesion." in text:
                value = 5
            elif "This subject has an Chronic brain lesion characterized by hemorrhage." in text:
                value = 6 
            elif "This subject has an Chronic brain lesion characterized by ICH." in text:
                value = 7
            elif "This subject has an Chronic brain lesion characterized by infarction." in text:
                value = 8
            elif "This subject has an Old brain lesion." in text:
                value = 5
            elif "This subject has an Old brain lesion characterized by hemorrhage." in text:
                value = 6
            elif "This subject has an Old brain lesion characterized by ICH." in text:
                value = 7
            elif "This subject has an Old brain lesion characterized by infarction." in text:
                value = 8
            else: 
                value = -1
            """
            """
            if text in "This subject does not have any brain lesion." :
                value = 0 
            elif text in "This subject has an Acute brain lesion.":
                value = 1
            elif text in "This subject has an Recent brain lesion.":
                value = 1
            elif text in "This subject has an Chronic brain lesion.":
                value = 2
            elif text in "This subject has an Old brain lesion.":
                value = 2
            else: 
                value = -1
            """
            if stage == 1:
                if "Yes" in text or "yes" in text: 
                    value = 0 
                elif "No" in text or "no" in text:
                    value = 1
                else: 
                    # other value
                    value = 2
            elif stage == 2: 
                if "Chronic" in text:
                    value = 0 
                elif "Acute" in text: 
                    value = 1
                else: 
                    # other value
                    value = 2
            elif stage == 3:
                if "Infarction" in text:
                    value = 0 
                elif "Intracerebral hemorrhage" in text:
                    value = 1 
                elif "Hemorrhage" in text: 
                    value = 2
                else: 
                    # other value
                    value = 3
                    
                
            return value  

        assert type(target) == type(pred) == list 
        assert len(target) == len(pred)
        
        target_list = [] 
        pred_list = []
        for target_text, pred_text in zip(target, pred): 
            target_value = _one_hot_encoder(target_text, stage)
            target_list.append(target_value)
            pred_value = _one_hot_encoder(pred_text, stage)
            pred_list.append(pred_value)
            
            #print(f"[DEBEG] GT(Label): ({target_value}){target_text} PRED(Label): ({pred_value}){pred_text}")
        return accuracy_score(target_list, pred_list)



    def training_step(self, batch, batch_idx): 
        total_loss = 0
        scale_count = 0
        opt = self.optimizers()
        if self.current_epoch < self.stage2_start_epoch: 
            opt.zero_grad()
            # reformul input data structure
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
            
            self.manual_backward(stage1_loss)
            opt.step()
        
            #try: 
            #    torch.cuda.empty_cache()
            #    stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            #except: 
            stage1_acc = float("nan")
                
            total_loss += stage1_loss.item()
            scale_count += 1
            
            self.log_dict({
                "train/stage1/loss": stage1_loss.item(),
                'train/stage1/acc': stage1_acc,
                "train/total_loss": stage1_loss.item(),
                            }, #sync_dist=True
                          )
            
            if batch_idx % 50 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
        

        elif self.stage2_start_epoch <= self.current_epoch < self.stage3_start_epoch:
            orig_img = batch['image'].clone()
            ## stage 1 learning
            opt.zero_grad()
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
            
            self.manual_backward(stage1_loss)
            opt.step()
        
            #try: 
            #    torch.cuda.empty_cache()
            #    stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            #except: 
            stage1_acc = float("nan")
                
            total_loss += stage1_loss.item()
            scale_count += 1
                
            
            self.log_dict({
                    "train/stage1/loss": stage1_loss.item(),
                    'train/stage1/acc': stage1_acc,
                                }, #sync_dist=True
                              )
            if batch_idx % 50 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
            
            ## stage 2 learning
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer2"][i], range(len(batch['answer']["answer2"]))))
            if len(filter_idx) > 0: 
                opt.zero_grad()
                batch['image'] = batch['image'][filter_idx]
                batch['text_input'] = [batch['inst']["inst2"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer2"][i] for i in filter_idx]
                stage2_loss, loss_dict, stage2_pred = self.model(batch)
                
                self.manual_backward(stage2_loss)
                opt.step()
                #try: 
                #    torch.cuda.empty_cache()
                #    stage2_acc = self.summarize_model_performance(batch['text_output'], stage2_pred, stage=2)
                #except: 
                stage2_acc = float("nan")

                total_loss += stage2_loss.item()
                scale_count += 1
                
                self.log_dict({
                    'train/stage2/loss': stage2_loss.item(),
                    'train/stage2/acc': stage2_acc,
                                }, #sync_dist=True
                              )
                
            else: 
                stage2_loss = None
                stage2_acc = None
                stage2_pred = None
                
            
            if batch_idx % 50 == 0: 
                print(f"\nstage2/ACC:{stage2_acc}\n    stage2/GT: {batch['answer']['answer2'][:4]}\n    stage2 PRED: {stage2_pred}")      
            torch.cuda.empty_cache()  


        else: 
            orig_img = batch['image'].clone()
            ## stage 1 learning
            opt.zero_grad()
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
            
            self.manual_backward(stage1_loss)
            opt.step()
        
            #try: 
            #    torch.cuda.empty_cache()
            #    stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            #except: 
            stage1_acc = float("nan")
                
            total_loss += stage1_loss.item()
            scale_count += 1
            
            self.log_dict({
                    "train/stage1/loss": stage1_loss.item(),
                    'train/stage1/acc': stage1_acc,
                                }, #sync_dist=True
                              )
            
            if batch_idx % 50 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
            
            ## stage 2 learning
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer2"][i], range(len(batch['answer']["answer2"]))))
            if len(filter_idx) > 0: 
                opt.zero_grad()
                batch['image'] = batch['image'][filter_idx]
                batch['text_input'] = [batch['inst']["inst2"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer2"][i] for i in filter_idx]
                stage2_loss, loss_dict, stage2_pred = self.model(batch)
                
                self.manual_backward(stage2_loss)
                opt.step()
                
                #try: 
                #    torch.cuda.empty_cache()
                #    stage2_acc = self.summarize_model_performance(batch['text_output'], stage2_pred, stage=2)
                #except: 
                stage2_acc = float("nan")
                    
                total_loss += stage2_loss.item()
                scale_count += 1
                
                self.log_dict({
                    'train/stage2/loss': stage2_loss.item(),
                    'train/stage2/acc': stage2_acc,
                                }, #sync_dist=True
                              )
            else: 
                stage2_loss = None
                stage2_acc = None
                stage2_pred = None
            
            if batch_idx % 50 == 0: 
                print(f"\nstage2/ACC:{stage2_acc}\n    stage2/GT: {batch['answer']['answer2'][:4]}\n    stage2 PRED: {stage2_pred}")
            torch.cuda.empty_cache()
                
            ## stage 3 learning
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer3"][i], range(len(batch['answer']["answer3"]))))
            if len(filter_idx) > 0: 
                opt.zero_grad()
                batch['image'] = orig_img[filter_idx]
                batch['text_input'] = [batch['inst']["inst3"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer3"][i] for i in filter_idx]
                stage3_loss, loss_dict, stage3_pred = self.model(batch)
                
                self.manual_backward(stage3_loss)
                opt.step()
                 
                #try: 
                #    torch.cuda.empty_cache()
                #    stage3_acc = self.summarize_model_performance(batch['text_output'], stage3_pred, stage=3)
                #except: 
                stage3_acc = float("nan")
                    
                total_loss += stage3_loss.item()
                scale_count += 1
                    
                self.log_dict({
                    'train/stage3/loss': stage3_loss.item(),
                    'train/stage3/acc': stage3_acc,
                }, #sync_dist=True
                              )
            else: 
                stage3_loss = None
                stage3_acc = None
                stage3_pred = None        
            
            if batch_idx % 50 == 0: 
                print(f"\nstage3/ACC:{stage3_acc}\n    stage3/GT: {batch['answer']['answer3'][:4]}\n    stage3 PRED: {stage3_pred}")
            torch.cuda.empty_cache()
            
        self.log_dict({
            "train/total_loss": total_loss / scale_count,
                        }, #sync_dist=True
                        )
            
        torch.cuda.empty_cache()
        
     

    def validation_step(self,batch, batch_idx): 
        torch.cuda.empty_cache()
        total_loss = 0
        scale_count = 0
        if self.current_epoch < self.stage2_start_epoch: 
            # reformul input data structure
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
            try: 
                stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            except: 
                stage1_acc = float("nan")
            
            total_loss += stage1_loss.item()
            scale_count += 1
                
            self.log_dict({
                "val/stage1/loss": stage1_loss.item(),
                'val/stage1/acc': stage1_acc,
            }, #sync_dist=True
                          )
            
            if batch_idx % 10 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
        
        
        elif self.stage2_start_epoch <= self.current_epoch < self.stage3_start_epoch:
            ## stage 1 learning
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
        
            try: 
                stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            except: 
                stage1_acc = float("nan")
            
            total_loss += stage1_loss.item()
            scale_count += 1
            
            self.log_dict({
                "val/stage1/loss": stage1_loss.item(),
                'val/stage1/acc': stage1_acc,
            }, #sync_dist=True
                              )
            if batch_idx % 10 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
                
                
            ## stage 2 learning    
            orig_img = batch['image'].clone()
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer2"][i], range(len(batch['answer']["answer2"]))))
            if len(filter_idx) > 0: 
                batch['image'] = batch['image'][filter_idx]
                batch['text_input'] = [batch['inst']["inst2"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer2"][i] for i in filter_idx]
                stage2_loss, loss_dict, stage2_pred = self.model(batch)
                
                try: 
                    stage2_acc = self.summarize_model_performance(batch['text_output'], stage2_pred, stage=2)
                except: 
                    stage2_acc = float("nan")
                    
                total_loss += stage2_loss.item()
                scale_count += 1
                
                self.log_dict({
                    'val/stage2/loss': stage2_loss.item(),
                    'val/stage2/acc': stage2_acc,
                }, #sync_dist=True
                              )
            else: 
                stage2_loss = None
                stage2_acc = None
                stage2_pred = None
            
            if batch_idx % 10 == 0: 
                print(f"\nstage2/ACC:{stage2_acc}\n    stage2/GT: {batch['answer']['answer2'][:4]}\n    stage2 PRED: {stage2_pred}")
            torch.cuda.empty_cache()
                
        
        else: 
            ## stage 1 learning
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
        
            try: 
                stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            except: 
                stage1_acc = float("nan")
            
            total_loss += stage1_loss.item()
            scale_count += 1
            
            self.log_dict({
                "val/stage1/loss": stage1_loss.item(),
                'val/stage1/acc': stage1_acc,
            }, #sync_dist=True
                              )
            if batch_idx % 10 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
                
                
            ## stage 2 learning    
            orig_img = batch['image'].clone()
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer2"][i], range(len(batch['answer']["answer2"]))))
            if len(filter_idx) > 0: 
                batch['image'] = batch['image'][filter_idx]
                batch['text_input'] = [batch['inst']["inst2"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer2"][i] for i in filter_idx]
                stage2_loss, loss_dict, stage2_pred = self.model(batch)
                
                try: 
                    stage2_acc = self.summarize_model_performance(batch['text_output'], stage2_pred, stage=2)
                except: 
                    stage2_acc = float("nan")
                
                total_loss += stage2_loss.item()
                scale_count += 1
                
                self.log_dict({
                    'val/stage2/loss': stage2_loss.item(),
                    'val/stage2/acc': stage2_acc,
                }, #sync_dist=True
                              )
            else: 
                stage2_loss = None
                stage2_acc = None
                stage2_pred = None
            
            if batch_idx % 10 == 0: 
                print(f"\nstage2/ACC:{stage2_acc}\n    stage2/GT: {batch['answer']['answer2'][:4]}\n    stage2 PRED: {stage2_pred}")
            torch.cuda.empty_cache()
                
            
            ## stage 3 learning
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer3"][i], range(len(batch['answer']["answer3"]))))
            if len(filter_idx) > 0: 
                batch['image'] = orig_img[filter_idx]
                batch['text_input'] = [batch['inst']["inst3"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer3"][i] for i in filter_idx]
                stage3_loss, loss_dict, stage3_pred = self.model(batch)
                
                try: 
                    stage3_acc = self.summarize_model_performance(batch['text_output'], stage3_pred, stage=3)
                except: 
                    stage3_acc = float("nan")
                
                total_loss += stage3_loss.item()
                scale_count += 1
                
                self.log_dict({
                    'val/stage3/loss': stage3_loss.item(),
                    'val/stage3/acc': stage3_acc,
                }, #sync_dist=True
                              )
            else:
                stage3_loss = None
                stage3_acc = None
                stage3_pred = None
                
            if batch_idx % 10 == 0: 
                print(f"\nstage3/ACC:{stage3_acc}\n    stage3/GT: {batch['answer']['answer3'][:4]}\n    stage3 PRED: {stage3_pred}")
            torch.cuda.empty_cache()


        self.log_dict({
            "val/total_loss": total_loss / scale_count,
                        }, #sync_dist=True
                        )
        #torch.cuda.empty_cache()


    def test_step(self,batch, batch_idx): 
        total_loss = 0
        scale_count = 0
        if self.current_epoch < self.stage2_start_epoch: 
            # reformul input data structure
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
            try: 
                stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            except: 
                stage1_acc = float("nan")
            
            total_loss += stage1_loss.item()
            scale_count += 1
                
            self.log_dict({
                "test/stage1/loss": stage1_loss.item(),
                'test/stage1/acc': stage1_acc,
            }, #sync_dist=True
                          )
            
            if batch_idx % 10 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
        
        
        elif self.stage2_start_epoch <= self.current_epoch < self.stage3_start_epoch:
            ## stage 1 learning
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
        
            try: 
                stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            except: 
                stage1_acc = float("nan")
            
            total_loss += stage1_loss.item()
            scale_count += 1
            
            self.log_dict({
                "test/stage1/loss": stage1_loss.item(),
                'test/stage1/acc': stage1_acc,
            }, #sync_dist=True
                              )
            if batch_idx % 10 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
                
                
            ## stage 2 learning    
            orig_img = batch['image'].clone()
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer2"][i], range(len(batch['answer']["answer2"]))))
            if len(filter_idx) > 0: 
                batch['image'] = batch['image'][filter_idx]
                batch['text_input'] = [batch['inst']["inst2"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer2"][i] for i in filter_idx]
                stage2_loss, loss_dict, stage2_pred = self.model(batch)
                
                try: 
                    stage2_acc = self.summarize_model_performance(batch['text_output'], stage2_pred, stage=2)
                except: 
                    stage2_acc = float("nan")
                    
                total_loss += stage2_loss.item()
                scale_count += 1
                
                self.log_dict({
                    'test/stage2/loss': stage2_loss.item(),
                    'test/stage2/acc': stage2_acc,
                }, #sync_dist=True
                              )
            else: 
                stage2_loss = None
                stage2_acc = None
                stage2_pred = None
            
            if batch_idx % 10 == 0: 
                print(f"\nstage2/ACC:{stage2_acc}\n    stage2/GT: {batch['answer']['answer2'][:4]}\n    stage2 PRED: {stage2_pred}")
            torch.cuda.empty_cache()
                
        
        else: 
            ## stage 1 learning
            batch['text_input'] = batch['inst']["inst1"]
            batch['text_output'] = batch['answer']["answer1"]
            stage1_loss, loss_dict, stage1_pred = self.model(batch)
        
            try: 
                stage1_acc = self.summarize_model_performance(batch['text_output'], stage1_pred, stage=1)
            except: 
                stage1_acc = float("nan")
            
            total_loss += stage1_loss.item()
            scale_count += 1
            
            self.log_dict({
                "test/stage1/loss": stage1_loss.item(),
                'test/stage1/acc': stage1_acc,
            }, #sync_dist=True
                              )
            if batch_idx % 10 == 0: 
                print(f"\nstage1/ACC:{stage1_acc}\n    stage1/GT: {batch['answer']['answer1'][:4]}\n    stage1 PRED: {stage1_pred}")
            torch.cuda.empty_cache()
                
                
            ## stage 2 learning    
            orig_img = batch['image'].clone()
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer2"][i], range(len(batch['answer']["answer2"]))))
            if len(filter_idx) > 0: 
                batch['image'] = batch['image'][filter_idx]
                batch['text_input'] = [batch['inst']["inst2"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer2"][i] for i in filter_idx]
                stage2_loss, loss_dict, stage2_pred = self.model(batch)
                
                try: 
                    stage2_acc = self.summarize_model_performance(batch['text_output'], stage2_pred, stage=2)
                except: 
                    stage2_acc = float("nan")
                
                total_loss += stage2_loss.item()
                scale_count += 1
                
                self.log_dict({
                    'test/stage2/loss': stage2_loss.item(),
                    'test/stage2/acc': stage2_acc,
                }, #sync_dist=True
                              )
            else: 
                stage2_loss = None
                stage2_acc = None
                stage2_pred = None
            
            if batch_idx % 10 == 0: 
                print(f"\nstage2/ACC:{stage2_acc}\n    stage2/GT: {batch['answer']['answer2'][:4]}\n    stage2 PRED: {stage2_pred}")
            torch.cuda.empty_cache()
                
            
            ## stage 3 learning
            filter_idx = list(filter(lambda i: "nan" not in  batch["answer"]["answer3"][i], range(len(batch['answer']["answer3"]))))
            if len(filter_idx) > 0: 
                batch['image'] = orig_img[filter_idx]
                batch['text_input'] = [batch['inst']["inst3"][i] for i in filter_idx]
                batch['text_output'] = [batch['answer']["answer3"][i] for i in filter_idx]
                stage3_loss, loss_dict, stage3_pred = self.model(batch)
                
                try: 
                    stage3_acc = self.summarize_model_performance(batch['text_output'], stage3_pred, stage=3)
                except: 
                    stage3_acc = float("nan")
                
                total_loss += stage3_loss.item()
                scale_count += 1
                
                self.log_dict({
                    'test/stage3/loss': stage3_loss.item(),
                    'test/stage3/acc': stage3_acc,
                }, #sync_dist=True
                              )
            else:
                stage3_loss = None
                stage3_acc = None
                stage3_pred = None
                
            if batch_idx % 10 == 0: 
                print(f"\nstage3/ACC:{stage3_acc}\n    stage3/GT: {batch['answer']['answer3'][:4]}\n    stage3 PRED: {stage3_pred}")
            torch.cuda.empty_cache()

    
        self.log_dict({
            "test/total_loss": total_loss / scale_count,
                        }, #sync_dist=True
                        )
        torch.cuda.empty_cache()


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
                from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam
                optim = DeepSpeedCPUAdam(filter(lambda p: p.requires_grad, self.parameters()), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
                #optim = DeepSpeedCPUAdam(self.parameters(), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
                pass
            else:
                optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
                #optim = torch.optim.AdamW(self.parameters(), lr= self.learning_rate, weight_decay=self.hparams.training_parameters.weight_decay)
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




