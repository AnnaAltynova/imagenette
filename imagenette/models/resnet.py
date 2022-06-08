# TODO 
# init weights
# SE
# resnext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
 

class ConvBlock(nn.Module):
    def __init__(self, kernel_size, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x    

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Identity()
        stride = 1
        
        if in_channels != out_channels:
            stride = 2
            self.projection = ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=stride)
            
        self.conv1 = ConvBlock(kernel_size=3, in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = ConvBlock(kernel_size=3, in_channels=out_channels, out_channels=out_channels)
        
        
    def forward(self, x):
        identity = x
        f = self.conv1(x)
        f = F.relu(f)
        f = self.conv2(f) 
        identity = self.projection(identity) 
        f = f + identity
        f = F.relu(f)
        return f
    

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False, squeeze_ratio=4):
        super().__init__()
        self.projection = nn.Identity()
        stride = 1
            
        if downsample:
            stride = 2
        if in_channels < out_channels:
            self.projection = ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=stride)        
        
        squeeze_channels = in_channels // squeeze_ratio
        
        self.conv1 = ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=squeeze_channels, stride=stride)
        self.conv2 = ConvBlock(kernel_size=3, in_channels=squeeze_channels, out_channels=squeeze_channels, stride=1)
        self.conv3 = ConvBlock(kernel_size=1, in_channels=squeeze_channels, out_channels=out_channels, stride=1)
        
        
    def forward(self, x):
        identity = x
        f = self.conv1(x)
        f = F.relu(f)
        f = self.conv2(f)
        f = F.relu(f)
        f = self.conv3(f)
        identity = self.projection(identity)
        f = f + identity
        f = F.relu(f)
        return f 
   

class ResnetDBlock(BottleneckBlock):
    def __init__(self, in_channels, out_channels, downsample=False, squeeze_ratio=4):
        super().__init__(in_channels=in_channels, out_channels=out_channels, downsample=downsample, squeeze_ratio=squeeze_ratio)
        projection_layers = [nn.Identity()]
        stride = 1
            
        if in_channels < out_channels:
            projection_layers.append(ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=out_channels, stride=1))
            
        if downsample:
            stride = 2
            projection_layers[0] = nn.AvgPool2d(kernel_size=2, stride=stride)
            
        self.projection = nn.Sequential(*projection_layers)
        
        squeeze_channels = in_channels // squeeze_ratio
        
        self.conv1 = ConvBlock(kernel_size=1, in_channels=in_channels, out_channels=squeeze_channels, stride=1)
        self.conv2 = ConvBlock(kernel_size=3, in_channels=squeeze_channels, out_channels=squeeze_channels, stride=stride)
        self.conv3 = ConvBlock(kernel_size=1, in_channels=squeeze_channels, out_channels=out_channels, stride=1)
        
    
class Resnet(nn.Module):
    def __init__(self, num_layers=18, in_channels=3, out_dim=10):
        super().__init__()
        if num_layers == 18:
            self.blocks_cnt = (2, 2, 2, 2)
        elif num_layers == 9:
            self.blocks_cnt = (1, 1, 1, 1)
        elif num_layers in (34, 50):
            self.blocks_cnt = (3, 4, 6, 3)
        else:
            print(f'resnet with num_layers {num_layers} is not implemented, building with 18 layers')
            num_layers = 18
            
        self.num_layers = num_layers
        self.in_channels = 64 
        
        if self.num_layers >= 50:
            self.blocks_out_channels = (256, 512, 1024, 2048)
            self.resnet_layers = self._make_bottleneck_layers()
        else:
            self.blocks_out_channels = (64, 128, 256, 512)
            self.resnet_layers = self._make_basic_layers()
            
        self.conv = ConvBlock(kernel_size=7, in_channels=in_channels, out_channels=64, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=self.blocks_out_channels[-1], out_features=out_dim)
        
    
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.resnet_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
        
        
    def _add_basic_layer(self, out_channels, n_blocks):
        blocks = []
        for i in range(n_blocks):
            blocks.append(ResnetBlock(in_channels=self.in_channels, out_channels=out_channels))
            self.in_channels = out_channels 
        return nn.Sequential(*blocks)
    
    
    def _add_bottleneck_layer(self, out_channels, n_blocks, downsample, squeeze_ratio):  
        blocks = []
        for i in range(n_blocks):
            blocks.append(BottleneckBlock(in_channels=self.in_channels, out_channels=out_channels,
                                          downsample=downsample, squeeze_ratio=squeeze_ratio))
            self.in_channels = out_channels 
            downsample = False
            squeeze_ratio = 4
        return nn.Sequential(*blocks)

    
    def _make_basic_layers(self):
        layers = []
        for i in range(4):
            layers.append(self._add_basic_layer(out_channels=self.blocks_out_channels[i], n_blocks=self.blocks_cnt[i]))
        return nn.Sequential(*layers)
    
    
    def _make_bottleneck_layers(self):
        layers = []
        downsample = False
        squeeze_ratio = 1
        for i in range(4):
            layers.append(self._add_bottleneck_layer(out_channels=self.blocks_out_channels[i], n_blocks=self.blocks_cnt[i],
                                                     downsample=downsample, squeeze_ratio=squeeze_ratio))
            downsample = True
            squeeze_ratio = 2
        return nn.Sequential(*layers)
    
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    

        
        
class ResnetD(Resnet):
    def __init__(self, in_channels=3, out_dim=10):
        super().__init__(num_layers=50, in_channels=in_channels, out_dim=out_dim)
        
    
    def _add_bottleneck_layer(self, out_channels, n_blocks, downsample, squeeze_ratio):         
        blocks = []
        for i in range(n_blocks):
            blocks.append(ResnetDBlock(in_channels=self.in_channels, out_channels=out_channels,
                                          downsample=downsample, squeeze_ratio=squeeze_ratio))
            self.in_channels = out_channels 
            downsample = False
            squeeze_ratio = 4
        return nn.Sequential(*blocks)

        