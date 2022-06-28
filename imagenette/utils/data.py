import pandas as pd
import numpy as np
from typing import Tuple
import PIL 
import os 
import pickle as pkl

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as T

import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2

from PIL import Image
import matplotlib._color_data as mcd
import cv2
import ipdb
import json



_NUMERALS = '0123456789abcdefABCDEF'
_HEXDEC = {v: int(v, 16) for v in (x+y for x in _NUMERALS for y in _NUMERALS)}
LOWERCASE, UPPERCASE = 'x', 'X'



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



class AdeDataset(Dataset):
    def __init__(self, transform=None, num_classes=151, mode='training', 
                 dataset_path='raw/ade2016/ADEChallengeData2016'):
        super().__init__()
        self.path_to_images = os.path.join(dataset_path, f'images/{mode}')
        self.path_to_masks = os.path.join(dataset_path, f'annotations/{mode}')
        self.images_path_collection = os.listdir(self.path_to_images)
        self.images_path_collection.remove('.ipynb_checkpoints')
        self.annotations_path_collection = os.listdir(self.path_to_masks)
        self.mode = mode
        self.transform = transform
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.images_path_collection)
        
    def __getitem__(self, idx):
        image_path = os.path.join(self.path_to_images, self.images_path_collection[idx])
        mask_path = os.path.join(self.path_to_masks, self.images_path_collection[idx].replace('.jpg', '.png'))
        # image = Image.open(image_path).convert('RGB')
        # mask = Image.open(mask_path)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        else:
            image = torch.from_numpy(np.array(image)).float()
        mask = torch.from_numpy(np.array(mask)).long()
        return image, mask
        
        
def get_ade_norm_stats():
    dataset = AdeDataset(transform=None, mode='training')
    means = []
    stds = []
    for i, (image, mask) in enumerate(dataset):
        if i == 1500:
            break
        means.append((image[..., 0].mean(), image[..., 1].mean(), image[..., 2].mean()))
        stds.append((image[..., 0].std(), image[..., 1].std(), image[..., 2].std()))

    mean, std = np.mean(means, axis=0), np.mean(stds, axis=0)
    return mean, std 


def get_ade_transforms(mean=[125, 119, 110], std=[57, 57, 61], resize=None):
    if mean is None:
        mean, std = get_ade_norm_stats()
    transforms = []
    if resize is not None:
        transforms.append(A.SmallestMaxSize(resize))
        transforms.append(A.CenterCrop(resize, resize))
    transforms.append(A.Normalize(mean=mean, std=std))
    transforms.append(ToTensorV2())
    transform = A.Compose(transforms)
    return transform 
