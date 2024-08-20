import os 
import numpy as np 
import pandas as pd 
import torch

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from monai.data import NibabelReader
from monai.transforms import LoadImage, Randomizable, apply_transform, AddChannel, Compose, RandRotate90, Resize, NormalizeIntensity, Flip, ToTensor, RandAxisFlip, RandAffine
from monai.utils import MAX_SEED, get_seed

from utils.utils import to_3tuple

class Text_Image_Dataset(Dataset, Randomizable): 
    def __init__(self, 
                 image_files=None, 
                 image_transform=None, 
                 reports_text=None, 
                 label_text=None, 
                 add_context=False, 
                 sex_text=None, 
                 age_text=None,
                 ):
        """
        Training Stage 1 
        - predict whether the subject has brain lesion or not 
        - if the subject has brain lesion, also predict injury time 
        self.quest = "Question 1: Provide 'yes' or 'no' answer as to whether this subject has a brain lesion. If 'yes' to Question 1, provide answer to Question 2. If 'no' to Question 1, no further response is needed. Question 2:If this subject has a brain lesion, choose the most appropriate option among 'acute', 'chronic', 'recent', and 'old'."
        Training Stage 2
        - predict whether the subject has brain lesion or not 
        - if the subject has brain lesion, also predict injury time 
        - if the subject has brain lesion, also predict injury time 
        self.quest = "Question 1: Provide 'yes' or 'no' answer as to whether this subject has a brain lesion. If 'yes' to Question 1, provide answer to Question 2. If 'no' to Question 1, no further response is needed. Question 2:If this subject has a brain lesion, choose the most appropriate option among 'acute', 'chronic', 'recent', and 'old'. Also, if this subject has a brain lesion, choose the most appropriate option among 'infarction', 'hemorrage', and 'ICH'."
        """
        self.image_files = image_files
        self.image_transform = image_transform
        self.reports_text = reports_text
        self.label_text = label_text
        self.add_context = add_context
        if self.add_context: 
            assert sex_text is not None and age_text is not None
        self.sex_text = sex_text
        self.age_text = age_text
        self.priming = 'You are a neurologist and now you are analyzing T2-weighted and FLAIR images from subjects who may be diagnosed with stroke. '
        self.quest = "Assess the subject's brain health status by evaluating the presence of any brain lesions, and if present, provide information on their injury time and characteristics"
        self.qna_template = self.quest + "Answer: "
        self.ans_template = "This subject will be diagnosed with"
        self.image_loader = LoadImage(reader=None, image_only=True, dtype=np.float32)    # use default reader of LoadImage
        self.set_random_state(seed=get_seed())
        self._seed = 0

    def randomize(self, data=None) -> None: 
        self._seed = self.R.randint(MAX_SEED, dtype='uint32')

    def __transform_image__(self, image_file):
        image = self.image_loader(image_file)
        if self.image_transform is not None: 
            if isinstance(self.image_transform, Randomizable): 
                self.image_transform.set_random_state(seed=self._seed)
            image = apply_transform(self.image_transform, image, map_items=False)
            image = torch.tensor(image)
        return image
    
    def __transform_text__(self, reports, label, add_context=False, sex=None, age=None):
        text = reports.replace("_x000D_\n", "")  # remove unnecessary spaces from text
        answer = self.__transform_text_label_(label)
        #answer = f'{self.ans_template} {label}.'
        inst = self.priming 
                
        if add_context:
            if sex == 'F': 
                context = f'This subject is {age}-years old female. '
            elif sex == 'M': 
                context = f'This subject is {age}-years old male. '
            text = context + text 
            inst = inst + context 
        
        inst = inst + text + self.quest
        text = text + answer 
        
        return text, inst, answer, label
    

    def __transform_text_label_(self, label): 
        _, lesion, injury_time, character, _ = label.split("\n")
        """
        if lesion.split(" ")[-1] == "Y": 
            answer1 = "Answer 1: yes"
            answer2 = f"Answer 2: {injury_time.split(' ')[-1]} and {character.split(' ')[-1]}." 
        elif lesion.split(" ")[-1] == "N":
            answer1 = "Answer 1: no"
            answer2 = "" 
        answer = f"{answer1}. {answer2}"
        """
        if lesion.split(" ")[-1] == "Y": 
            if character.split(' ')[-1] == "nan":
                answer = f"This subject has an {injury_time.split(' ')[-1]} brain lesion."
            else: 
                answer = f"This subject has an {injury_time.split(' ')[-1]} brain lesion characterized by {character.split(' ')[-1]}."
        elif lesion.split(" ")[-1] == "N":
            answer = "This subject does not have any brain lesion."
        return answer 



    def __len__(self) -> int: 
        return len(self.image_files)
    
    def __getitem__(self, index:int): 
        """
        output: {"image": torch.tensor, "text": str, "inst": str, "answer": str, "label": str}
        """
        image = self.__transform_image__(image_file=self.image_files[index])
        if self.add_context:
            text, inst, answer, label = self.__transform_text__(reports=self.reports_text[index], label=self.label_text[index], add_context=True, sex=self.sex_text[index], age=self.age_text[index])
        else: 
            text, inst, answer, label = self.__transform_text__(reports=self.reports_text[index], label=self.label_text[index], add_context=False)
        return {
                'image': image,
                'text': text,
                'inst': inst,
                'answer': answer,
                'label': label
                }


class  Text_Image_DataModule(pl.LightningDataModule):
    def __init__(self, config_dataset:dict = None, batch_size=1):
        self.save_hyperparameters(config_dataset)
        self.img_dir = self.hparams.img_dir
        self.meta_dir = self.hparams.meta_dir
        # checking whether the data directory is correctly set 
        self.batch_size=batch_size

        # setup dataset 
        #self.setup()
        self.prepare_data_per_node=True
        self.allow_zero_length_dataloader_with_multiple_devices = True



    def define_image_augmentation(self, mode='train'):
        img_size = to_3tuple(self.hparams.img_size)
        if mode == 'train':
            transform = Compose([AddChannel(),
                               Resize(img_size),
                               RandAxisFlip(prob=0.5),
                               NormalizeIntensity()])
        elif mode == 'eval': 
            transform = Compose([AddChannel(),
                             Resize(img_size),
                             NormalizeIntensity()])
        return transform



    def get_dataset(self, img_dir=None, meta_dir=None): 
        ## loading meta data 
        if meta_dir is None: 
            raise ValueError("YOU SHOULD SPECIFY A DIRECTORY OF META DATA IN a '/config/*.yaml' FILE")
        elif meta_dir.find('.csv') != -1:
            meta_data = pd.read_csv(meta_dir)
        elif meta_dir.find('.xlsx') != -1:
            meta_data = pd.read_excel(meta_dir)
        else: 
            NotImplementedError("This code need only meta data file of '.csv' or '.xlsx' format.")
        
        ## randomly assign subjects into train/val/test split
        #total_subj = 20
        total_subj = len(meta_data)
        shuffle_idx = np.arange(total_subj)
        np.random.shuffle(shuffle_idx)

        assert self.hparams.train_size + self.hparams.val_size + self.hparams.test_size == 1
        train_subj = int(self.hparams.train_size * total_subj)
        val_subj = int(self.hparams.val_size * total_subj)
        test_subj = total_subj - val_subj

        ## split image 
        train_images = [os.path.join(img_dir, img) for img in meta_data['BET_output_path'].values[shuffle_idx[:train_subj]]]
        val_images = [os.path.join(img_dir, img) for img in meta_data['BET_output_path'].values[shuffle_idx[train_subj:train_subj+val_subj]]]
        test_images = [os.path.join(img_dir, img) for img in meta_data['BET_output_path'].values[shuffle_idx[train_subj+val_subj:]]]

        ## split text 
        train_text = meta_data['reports'].values[shuffle_idx[:train_subj]]
        val_text = meta_data['reports'].values[shuffle_idx[train_subj:train_subj+val_subj]]
        test_text = meta_data['reports'].values[shuffle_idx[train_subj+val_subj:]]

        ## split label 
        train_label = meta_data['label'].values[shuffle_idx[:train_subj]]
        val_label = meta_data['label'].values[shuffle_idx[train_subj:train_subj+val_subj]]
        test_label = meta_data['label'].values[shuffle_idx[train_subj+val_subj:]]
        
        if self.hparams.add_context: 
            ## split sex 
            train_sex = meta_data['sex'].values[shuffle_idx[:train_subj]]
            val_sex = meta_data['sex'].values[shuffle_idx[train_subj:train_subj+val_subj]]
            test_sex = meta_data['sex'].values[shuffle_idx[train_subj+val_subj:]]

            ## split age 
            train_age = meta_data['age'].values[shuffle_idx[:train_subj]]
            val_age = meta_data['age'].values[shuffle_idx[train_subj:train_subj+val_subj]]
            test_age = meta_data['age'].values[shuffle_idx[train_subj+val_subj:]]
            
            ## prepare dataset    
            train_dataset = Text_Image_Dataset(image_files=train_images, image_transform=self.define_image_augmentation(mode='train'), reports_text=train_text, label_text=train_label,  add_context=True, sex_text=train_sex, age_text=train_age)
            val_dataset = Text_Image_Dataset(image_files=val_images, image_transform=self.define_image_augmentation(mode='eval'), reports_text=val_text, label_text=val_label,  add_context=True, sex_text=val_sex, age_text=val_age)
            test_dataset = Text_Image_Dataset(image_files=test_images, image_transform=self.define_image_augmentation(mode='eval'), reports_text=test_text, label_text=test_label,  add_context=True, sex_text=test_sex, age_text=test_age)
        else: 
            ## prepare dataset    
            train_dataset = Text_Image_Dataset(image_files=train_images, image_transform=self.define_image_augmentation(mode='train'), reports_text=train_text, label_text=train_label,  add_context=False, sex_text=None, age_text=None)
            val_dataset = Text_Image_Dataset(image_files=val_images, image_transform=self.define_image_augmentation(mode='eval'), reports_text=val_text, label_text=val_label,  add_context=False, sex_text=None, age_text=None)
            test_dataset = Text_Image_Dataset(image_files=test_images, image_transform=self.define_image_augmentation(mode='eval'), reports_text=test_text, label_text=test_label,  add_context=False, sex_text=None, age_text=None)
        return train_dataset, val_dataset, test_dataset

    def prepare_data(self):
        return
    
    def setup(self, stage=None): 
        train_dataset, val_dataset, test_dataset = self.get_dataset(img_dir=self.img_dir, meta_dir=self.meta_dir)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)


    def train_dataloader(self):
        return self.train_loader


    def val_dataloader(self):
        return self.val_loader


    def test_dataloader(self):
        return self.test_loader


    def predict_dataloader(self):
        NotImplementedError()
        pass 


