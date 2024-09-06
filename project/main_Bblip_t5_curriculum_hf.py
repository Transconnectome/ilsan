import datetime
import hashlib
import json 
from model.Bblip_t5_curriculum_hf import PatchEmbed
from omegaconf import OmegaConf

import torch

from dataset.dataset_curriculum import Text_Image_DataModule
from transformers import AutoTokenizer, Blip2ForConditionalGeneration, Trainer, TrainingArguments
from accelerate.utils import DistributedType


from utils.utils import lambda_fn
import functools
import os 
import wandb

import warnings
warnings.filterwarnings('ignore')

def __main__(): 
    ### make experiment ID 
    time_hash = datetime.datetime.now().time()
    hash_key = hashlib.sha1(str(time_hash).encode()).hexdigest()[:6]

    ### loading and checking trainer configuration file 
    config_dir = "./project/config/Brain_blip_t5_train_multi-gpu_ds-zero3-offload_hf_curriculum.yaml"
    config = OmegaConf.load(config_dir)
    with open(config_dir, 'r') as config_f: 
        f = json.load(config_f)

    ### setting logger 
    wandb.login(key=config.wandb.API_KEY)
    os.environ['WANDB_PROJECT'] = "ilsan"


    ### setting tokenizer and datamodul
    tokenizer = AutoTokenizer.from_pretrained(config.model.hf_name)
    DataModule = Text_Image_DataModule(config_dataset=config.dataset,batch_size=config.training_parameters.batch_size, tokenizer=tokenizer)
 

    ### setting model 
    model = Blip2ForConditionalGeneration.from_pretrained(config.model.hf_name)
    patch_embed_3d = PatchEmbed(
            img_size=config.dataset.img_size, 
            #patch_size=self.model.vision_model.embeddings.patch_embedding.kernel_size[0], 
            patch_size=18,
            in_chans=1, 
            embed_dim=int(model.vision_model.embeddings.patch_embedding.out_channels))
    setattr(model.vision_model, "embeddings", patch_embed_3d)

    # replace patchify layer and position embeddingof vision model
    for name, param in model.vision_model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
        if 'post_layernorm' in name:
            param.requires_grad = False
        if 'embeddings' in name: 
            param.requires_grad = True
        

    # freeze Qformer
    for name, param in model.named_parameters():
        if 'qformer' in name:
            param.requires_grad = False
        if 'language_projection' in name:
            param.requires_grad = False
                

    # freeze LLM
    for name, param in model.named_parameters():
        if 'language_model' in name:
            param.requires_grad = False
    #model.cuda()
    
    # set gradient checkpointing
    model.gradient_checkpointing_enable()

    training_args = TrainingArguments(
        output_dir='./hf_results',          # output directory
        report_to = 'wandb',
        deepspeed = './config/hf_deepspeed_zero3_offload.json',
        run_name = f'{hash_key}',
        num_train_epochs=config.pl_trainer.max_epochs,              # total number of training epochs
        #per_device_train_batch_size=config.training_parameters.batch_size,  # batch size per device during training
        #per_device_eval_batch_size=config.training_parameters.batch_size,   # batch size for evaluation
        do_train = True, 
        do_eval = True,
        eval_strategy="steps", 
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./hf_logs',            # directory for storing logs
        logging_steps=10,
        # arguments for reducing memory (+ deepspeed zero3 offload)
        bf16=True, 
        bf16_full_eval=True,
        gradient_checkpointint=True,
        diable_tqdm=False, 
    )
    
    ### using optimizer defined in deepspeed configuration file
    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    trainer = Trainer(
        args=training_args,
        model=model,
        train_dataset = DataModule.train_dataset,
        eval_dataset = DataModule.val_dataset,
    )

if __name__ == '__main__': 
    __main__()
