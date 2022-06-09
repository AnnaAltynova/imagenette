import pandas as pd
import numpy as np
from typing import Tuple
import PIL 
import os 

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data import DataLoader


def generate_labels(directory: str, out_path: str) -> None:
    """"make csv file with path to files and corresponding labels"""
    class_names = [name for name in os.listdir(directory) if name != '.ipynb_checkpoints']
    name_to_num = dict(zip(class_names, range(len(class_names))))
    labels = pd.DataFrame(columns=['filepath', 'class_name', 'class_num'])
    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        files = [name for name in os.listdir(class_dir) if name != '.ipynb_checkpoints']
        class_df = pd.DataFrame(data={'filepath': [os.path.join(class_dir, file) for file in files],
                                      'class_name': np.repeat(class_name, len(files)), 
                                      'class_num': np.repeat(name_to_num[class_name], len(files))}, 
                                index=range(len(files)))
        labels = labels.append(class_df, ignore_index= True)
    labels.to_csv(out_path, index=False) 
    return 


class ImagenetteDataset(Dataset):
    def __init__(self, labels_csv_path: str, transform: None):
        self.labels = pd.read_csv(labels_csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.labels) 
    
    def __getitem__(self, idx):
        img_path, label = self.labels.iloc[idx][['filepath', 'class_num']]
        image = PIL.Image.open(img_path)
        if image.mode != 'RGB':
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label
        
        
        
def get_norm_stats(data_path: str, size=224, batch_size=256) -> Tuple[float, float]:
    """compute mean and std on large batch to perform normalisation """
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(size), 
        T.CenterCrop(size)])
    train_dataset = ImagenetteDataset(data_path, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
    imgs, labels = next(iter(train_dataloader))
    mean = imgs[:, 0, :, :].mean(), imgs[:, 1, :, :].mean(), imgs[:, 2, :, :].mean()
    std = imgs[:, 0, :, :].std(), imgs[:, 1, :, :].std(), imgs[:, 2, :, :].std()
    return mean, std


def get_transforms(mean, std, size=224):
    transform = T.Compose([
        T.ToTensor(),
        T.Resize(size), 
        T.CenterCrop(size),
        T.Normalize(mean=mean, std=std),
    ])
    return transform 


def get_augmented_transforms(mean, std, size=224):
    train_transform = T.Compose([
        T.ToTensor(),
        T.Resize(size), 
        T.CenterCrop(size),
        T.Normalize(mean=mean, std=std),
        T.RandomHorizontalFlip(p=1),
        T.RandomRotation(degrees=90),
        T.RandomInvert(p=0.5)
        
    ])
    
    return train_transform
    