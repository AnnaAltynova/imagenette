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
    def __init__(self, num_classes=151, num_units_collection=(3, 6, 12, 8), k_factor=12, 
                 pretrained=False, ckpt_path=f'saves/ckpts3/densenet_staged_model.pt', use_dropout=False, dropout_prob=0.5):
        super().__init__()
        densenet_out = []  
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
        self.blockpool1 = nn.Sequential(*[densenet.layers['initial_block'], densenet.layers['blockpool1']])    # transition 1
        self.blockpool2 = densenet.layers['blockpool2']                                                        # transition 2
        self.blockpool3 = densenet.layers['blockpool3']                                                        # transition 3
        self.block4 = densenet.layers['block4']                                                                # DenseBlock4(BN-relu-conv)
        
        channels = densenet_out[2] + densenet_out[3]
        self.deconv3 = ConvBlock(stride=2, kernel_size=2, in_channels=channels, out_channels=channels)         # blockpool3 -> blockpool2 
        channels += densenet_out[1]
        self.deconv2 = ConvBlock(stride=2, kernel_size=2, in_channels=channels, out_channels=channels)         # blockpool2 -> blockpool1 
        channels += densenet_out[0]
        self.pointwise = ConvBlock(kernel_size=1, stride=1, in_channels=channels, out_channels=num_classes, deconv=False)
        self.drop = nn.Dropout(p=dropout_prob,  inplace=True) if use_dropout else nn.Identity() 
        self.classifier = ConvBlock(kernel_size=8, stride=8, in_channels=num_classes, out_channels=num_classes)
        
    
    def forward(self, x):
        inputs = x
        pool1 = self.blockpool1(x)
        pool2 = self.blockpool2(pool1)
        pool3 = self.blockpool3(pool2)
        
        x = self.block4(pool3)
        x = torch.cat([x, pool3], dim=1)
        x = self.deconv3(x)
        
        x = torch.cat([x, pool2], dim=1)
        x = self.deconv2(x)
        
        x = torch.cat([x, pool1], dim=1)
        x = self.pointwise(x)
        x = self.drop(x)
        x = self.classifier(x)
        
        return x
    