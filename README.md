# Settings 
## Setting meta data 
It is recommended to set the name of meta data file as "meta_data.xlsx".  
Some comlumn names of meta data should be changed as follows.
- 환자번호 -> subjectkey
- 성별 -> sex 
- 검사연령-> age 
- 판독소견 -> reports

## Setting img data 
image files should be structured as follows. 
```
/img_dir
    |-----{BET_output_path}.nii.gz
```


## Setting data directories 
Absolute directories of image file and meta data should be respectively added to "img_dir" and "meta_dir"  in ```./project/config/Brain_blip_v1_train_single_gpu.yaml```.

## Setting wandb service
API key of your wandb account should be added to "API_KEY" in ```./project/config/Brain_blip_v1_train_single_gpu.yaml``` 

# Example 
```
cd {usr_dir}/project

python3 main_Bblip_v1.py

```






# Training 
- **"DeepSpeed stage3 zero optimize"**
1. for efficient GPU memory, it is highly recommened to use **"DeepSpeed stage3 zero optimize"**
2. When using this, only **"bfloat16"** precision should be used since "Qwen" only support flash attention inference with "float16" or "bfloat16" and pytorch support autocast on cpu device with "bfloat16".

# Reference Paper 
1. BLIP-2: https://arxiv.org/abs/2301.12597
2. Instruct BLIP: https://arxiv.org/abs/2305.06500
3. MedBLIP: https://arxiv.org/abs/2305.10799


