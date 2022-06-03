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
        x = F.relu(x)
        return x
    
    
class Projection(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
    
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Identity()
        stride = 1
        
        if in_channels != out_channels:
            stride = 2
            self.projection = Projection(in_channels=in_channels, out_channels=out_channels, stride=stride)
            
        self.conv1 = ConvBlock(kernel_size=3, in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv2 = ConvBlock(kernel_size=3, in_channels=out_channels, out_channels=out_channels)
        
        
    def forward(self, x):
        f = self.conv1(x)
        f = self.conv2(f) 
        x = self.projection(x) 
        x = f + x
        x = F.relu(x)
        return x
    
    
class Resnet18(nn.Module):
    def __init__(self, in_channels=3, out_channels=10):
        super().__init__()
        self.conv = ConvBlock(kernel_size=7, in_channels=in_channels, out_channels=64, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.resblock1 = ResnetBlock(in_channels=64, out_channels=64)
        self.resblock2 = ResnetBlock(in_channels=64, out_channels=128)
        self.resblock3 = ResnetBlock(in_channels=128, out_channels=256)
        self.resblock4 = ResnetBlock(in_channels=256, out_channels=512)
        self.avgpool = nn.AvgPool2d(kernel_size=7)
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=512, out_features=out_channels)
        
    
    def forward(self, x):
        x = self.conv(x)
        x = self.maxpool(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
        