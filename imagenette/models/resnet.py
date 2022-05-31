import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

def resnet_brick(kernel_size, in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False), 
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )    


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = None
        stride=1
        if in_channels != out_channels:
            stride = 2
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), 
                nn.BatchNorm2d(out_channels)
            )
        self.brick1 = resnet_brick(kernel_size=3, in_channels=in_channels,
                                   out_channels=out_channels, stride=stride)
        self.brick2 = resnet_brick(kernel_size=3, in_channels=out_channels, out_channels=out_channels)
        
        
        
    def forward(self, x):
        f = self.brick1(x)
        f = self.brick2(f) 
        if self.projection is not None:
            x = self.projection(x) 
        x = f + x
        x = F.relu(x)
        return x
    
    
def Resnet18(in_channels=3, out_channels=10):
    
    return torch.nn.Sequential(
        # in: 224x224x3
        resnet_brick(kernel_size=7, in_channels=in_channels, out_channels=64, stride=2),
        # in: 112x112x64
        nn.MaxPool2d(kernel_size=2, stride=2),
        # in: 56x56x64
        ResnetBlock(64, 64),
        # in: 56x56x64
        ResnetBlock(64, 128),
        # in: 28x28x128
        ResnetBlock(128, 256),
        # in: 14x14x256
        ResnetBlock(256, 512),
        # in: 7x7x512
        nn.AvgPool2d(kernel_size=7),
        nn.Flatten(),
        nn.Linear(512, out_channels)
    )
        