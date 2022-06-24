import torch 
import torch.nn as nn
import torch.nn.functional as F

from models import DenseNet
from utils import load_model


class ConvBlock(nn.Sequential):
    # preserves densenet convblock order
        def __init__(self, stride=37, kernel_size=4, in_channels=10, out_channels=10, deconv=True):
            layers = []
            layers.append(nn.BatchNorm2d(num_features=in_channels))  
            layers.append(nn.ReLU(inplace=True))
            if deconv:
                layers.append(nn.ConvTranspose2d(stride=stride, kernel_size=kernel_size,
                                                 in_channels=in_channels, out_channels=out_channels, bias=False))
            else:
                layers.append(nn.Conv2d(stride=stride, kernel_size=kernel_size,
                                                 in_channels=in_channels, out_channels=out_channels, bias=False))
                                
            super().__init__(*layers)
            

class FCN(nn.Module):
    """FCN with DenseNet121 backbone"""
    def __init__(self, num_classes=151, num_units_collection=(3, 6, 12, 8), k_factor=16, 
                 pretrained=False, ckpt_path=f'saves/ckpts3/densenet121new_model.pt'):
        super().__init__()
        densenet_out = []  # densenet_blocks out_channels
        channels = k_factor * 2 
        for num_units in num_units_collection[:-1]:
            out_channels = (channels + k_factor * num_units) // 2
            densenet_out.append(out_channels)
            channels = densenet_out[-1]
            
        out_channels = (channels + k_factor * num_units_collection[-1])
        densenet_out.append(out_channels)
        
        densenet = DenseNet(num_units_collection=num_units_collection, k_factor=k_factor)
        if pretrained:
            _ = load_model(model=densenet, ckpt_path=ckpt_path)
        
        densenet_modules = nn.ModuleList(densenet.children()) 
        self.blockpool1 = nn.Sequential(*densenet_modules[:7])    # transition 1
        self.blockpool2 = nn.Sequential(*densenet_modules[7:10])  # transition 2
        self.blockpool3 = nn.Sequential(*densenet_modules[10:13]) # transition 3
        self.block4 = densenet_modules[13]                        # DenseBlock4(BN-relu-conv)
        
        self.deconv4 = ConvBlock(deconv=False, stride=1, kernel_size=1, 
                                 in_channels=densenet_out[3], out_channels=densenet_out[2]) # block4 -> blockpool3, 
        self.deconv3 = ConvBlock(stride=4, kernel_size=4, in_channels=densenet_out[2], out_channels=densenet_out[1]) # blockpool3 -> blockpool2 
        self.deconv2 = ConvBlock(stride=2, kernel_size=2, in_channels=densenet_out[1], out_channels=densenet_out[0]) # blockpool2 -> blockpool1 
        self.classifier = ConvBlock(kernel_size=8, stride=8, in_channels=densenet_out[0], out_channels=num_classes)
        
        
    def count_padding(self, target, x):
        height = (target.shape[2] - x.shape[2]) 
        width = (target.shape[3] - x.shape[3])
        pad = width // 2, width // 2 + width % 2, height // 2, height // 2 + height % 2
        return pad
    
        
    def forward(self, x):
        inputs = x
        x = self.blockpool1(x)
        pool1 = x
        x = self.blockpool2(x)
        pool2 = x
        x = self.blockpool3(x)
        pool3 = x
        x = self.block4(x)
        
        x = self.deconv4(x)
        x = x + pool3
        
        x = self.deconv3(x)
        x = F.pad(input=x, pad=self.count_padding(pool2, x))
        x = x + pool2
        
        x = self.deconv2(x)
        x = F.pad(input=x, pad=self.count_padding(pool1, x))
        x = x + pool1
        
        x = self.classifier(x)
        x = F.pad(input=x, pad=self.count_padding(inputs, x))
        return x
    