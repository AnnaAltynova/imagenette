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



def rgb(triplet):
    return _HEXDEC[triplet[0:2]], _HEXDEC[triplet[2:4]], _HEXDEC[triplet[4:6]]

def loadAde20K(file):
    fileseg = file.replace('.jpg', '_seg.png');
    with Image.open(fileseg) as io:
        seg = np.array(io);

    # Obtain the segmentation mask, bult from the RGB channels of the _seg file
    R = seg[:,:,0];
    G = seg[:,:,1];
    B = seg[:,:,2];
    ObjectClassMasks = (R/10).astype(np.int32)*256+(G.astype(np.int32));


    # Obtain the instance mask from the blue channel of the _seg file
    Minstances_hat = np.unique(B, return_inverse=True)[1]
    Minstances_hat = np.reshape(Minstances_hat, B.shape)
    ObjectInstanceMasks = Minstances_hat


    level = 0
    PartsClassMasks = [];
    PartsInstanceMasks = [];
    while True:
        level = level+1;
        file_parts = file.replace('.jpg', '_parts_{}.png'.format(level));
        if os.path.isfile(file_parts):
            with Image.open(file_parts) as io:
                partsseg = np.array(io);
            R = partsseg[:,:,0];
            G = partsseg[:,:,1];
            B = partsseg[:,:,2];
            PartsClassMasks.append((np.int32(R)/10)*256+np.int32(G));
            PartsInstanceMasks = PartsClassMasks
            # TODO:  correct partinstancemasks

            
        else:
            break

    objects = {}
    parts = {}

    attr_file_name = file.replace('.jpg', '.json')
    if os.path.isfile(attr_file_name):
        with open(attr_file_name, 'r') as f:
            input_info = json.load(f)

        contents = input_info['annotation']['object']
        instance = np.array([int(x['id']) for x in contents])
        names = [x['raw_name'] for x in contents]
        corrected_raw_name =  [x['name'] for x in contents]
        partlevel = np.array([int(x['parts']['part_level']) for x in contents])
        ispart = np.array([p>0 for p in partlevel])
        iscrop = np.array([int(x['crop']) for x in contents])
        listattributes = [x['attributes'] for x in contents]
        polygon = [x['polygon'] for x in contents]
        for p in polygon:
            p['x'] = np.array(p['x'])
            p['y'] = np.array(p['y'])

        objects['instancendx'] = instance[ispart == 0]
        objects['class'] = [names[x] for x in list(np.where(ispart == 0)[0])]
        objects['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 0)[0])]
        objects['iscrop'] = iscrop[ispart == 0]
        objects['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 0)[0])]
        objects['polygon'] = [polygon[x] for x in list(np.where(ispart == 0)[0])]


        parts['instancendx'] = instance[ispart == 1]
        parts['class'] = [names[x] for x in list(np.where(ispart == 1)[0])]
        parts['corrected_raw_name'] = [corrected_raw_name[x] for x in list(np.where(ispart == 1)[0])]
        parts['iscrop'] = iscrop[ispart == 1]
        parts['listattributes'] = [listattributes[x] for x in list(np.where(ispart == 1)[0])]
        parts['polygon'] = [polygon[x] for x in list(np.where(ispart == 1)[0])]

    return {'img_name': file, 'segm_name': fileseg,
            'class_mask': ObjectClassMasks, 'instance_mask': ObjectInstanceMasks, 
            'partclass_mask': PartsClassMasks, 'part_instance_mask': PartsInstanceMasks, 
            'objects': objects, 'parts': parts}


def plot_polygon(img_name, info, show_obj=True, show_parts=False):
    colors = mcd.CSS4_COLORS
    color_keys = list(colors.keys())
    all_objects = []
    all_poly = []
    if show_obj:
        all_objects += info['objects']['class']
        all_poly += info['objects']['polygon']
    if show_parts:
        all_objects += info['parts']['class']
        all_poly += info['objects']['polygon']

    img = cv2.imread(img_name)
    thickness = 5
    for it, (obj, poly) in enumerate(zip(all_objects, all_poly)):
        curr_color = colors[color_keys[it % len(color_keys)] ]
        pts = np.concatenate([poly['x'][:, None], poly['y'][:, None]], 1)[None, :]
        color = rgb(curr_color[1:])
        img = cv2.polylines(img, pts, True, color, thickness)
    return img



class ADE20K_Dataset(Dataset):
    def __init__(self, ids:list, transform=None, num_classes=3688, index_file='ADE20K_2021_17_01/index_ade20k.pkl', dataset_path='raw/ade20k'):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.transform = transform 
        with open('{}/{}'.format(dataset_path, index_file), 'rb') as f:
            index_ade20k = pkl.load(f) 
            index = {}
            index['filename'] = np.array(index_ade20k['filename'])[ids]
            index['folder'] = np.array(index_ade20k['folder'])[ids]
        self.index = index
            
    
    def __len__(self):
        return len(self.index['filename'])
    
    
    def __getitem__(self, idx):
        full_file_name = '{}/{}'.format(self.index['folder'][idx], self.index['filename'][idx])
        info = loadAde20K('{}/{}'.format(self.dataset_path, full_file_name))
        
        class_mask = info['class_mask']
        # class_mask = class_mask.to_sparse_csr()
        # class_mask = F.one_hot(class_mask, num_classes=self.num_classes)
        image = cv2.imread(info['img_name'])
        
        # if image.mode != 'RGB':
        #     image = image.convert("RGB")
        if self.transform is not None:
            transformed = self.transform(image=image, mask=class_mask)
            image, class_mask = transformed['image'], transformed['mask']
            image = image.float()
        return image, class_mask
        
        
def get_ade_transforms():
    transform = A.Compose([
        A.PadIfNeeded(min_height=512, min_width=512),
        A.RandomCrop(512, 512),
        ToTensorV2()
    ])
    return transform 
    