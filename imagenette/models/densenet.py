import torch
import torch.nn as nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__(*[
                            nn.BatchNorm2d(num_features=in_channels),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, bias=False,
                                      kernel_size=kernel_size, padding=padding, stride=stride)
                            ])    
    
    
class DenseBlock(nn.Module):
    def __init__(self, in_channels, k_factor, num_units, bottleneck=False):
        """
        k_factor: #output channels in each convblock
        num_units: #convblocks
        bottleneck: whether to use bottleneck convblocks
        
        """
        super().__init__()
        self.in_channels = in_channels
        self.num_units = num_units
        self.bottleneck = bottleneck
        self.dense_units = nn.ModuleList()
        
        
        for i in range(num_units):
            if bottleneck:
                pointwise = ConvBlock(kernel_size=1, in_channels=in_channels, out_channels= 4 * k_factor, padding=0) 
                in_channels = 4 * k_factor
            else: 
                pointwise = nn.Identity()
                
            self.dense_units.append(nn.Sequential(*[pointwise, 
                                                   ConvBlock(in_channels=in_channels,out_channels=k_factor)]))
            in_channels = self.in_channels + k_factor * (i + 1) 
        
    def forward(self, x):
        for unit in self.dense_units:
            out = unit(x)
            x = torch.cat([x, out], dim=1)        
        return x
              

class DenseNet(nn.Module):
    def __init__(self, num_units_collection=(6, 12, 24, 16), k_factor=32, transition_factor=0.5,
                 bottleneck=True, in_channels=3, num_classes=10):
        """
        num_units_collection: number of convblocks in each dense block
        k_factor: #output channels in each convblock
        transition_factor: reduce #channels between blocks
        bottleneck: whether to use bottleneck convblocks
        
        returns list of outputs of transition layers
        
        """
        super().__init__()
        self.in_channels = in_channels
        
        self.layers = nn.ModuleDict()
        initial_block = nn.Sequential(*[nn.Conv2d(kernel_size=7, in_channels=3, 
                                                out_channels=k_factor * 2, stride=2, padding=3, 
                                                bias=False),
                                      nn.BatchNorm2d(num_features=k_factor * 2), 
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2, padding=1)])
        
        self.layers['initial_block'] = initial_block
        in_channels = k_factor * 2
        
        for i, num_units in enumerate(num_units_collection[:-1]):
            dense_block = DenseBlock(in_channels=in_channels, k_factor=k_factor,
                                     num_units=num_units, bottleneck=bottleneck)
            in_channels += num_units * k_factor
            blockpool = nn.Sequential(*[
                        dense_block, 
                        ConvBlock(kernel_size=1, in_channels=in_channels, 
                                  out_channels=int(in_channels * transition_factor), stride=1, padding=0),
                        nn.AvgPool2d(kernel_size=2, stride=2)])
            
            self.layers[f'blockpool{i + 1}'] = blockpool
            in_channels = int(in_channels * transition_factor)
            
        
        dense_block = DenseBlock(in_channels=in_channels, k_factor=k_factor, 
                                 num_units=num_units_collection[-1], bottleneck=bottleneck)
        in_channels += num_units_collection[-1] * k_factor 
        
        self.layers[f'block{len(num_units_collection)}'] = dense_block
        
        head = nn.Sequential(*[
                       nn.BatchNorm2d(num_features=in_channels), 
                       nn.AvgPool2d(kernel_size=7),
                       nn.Flatten(),
                       nn.Linear(in_features=in_channels, out_features=num_classes)
        ])
        self.layers['head'] = head
        
        
    def forward(self, x):
        x = self.layers['initial_block'](x)
        pool1 = self.layers['blockpool1'](x)     # 30
        pool2 = self.layers['blockpool2'](pool1) # 51
        pool3 = self.layers['blockpool3'](pool2) # 97
        block4 = self.layers['block4'](pool3)    # 193
        out = self.layers['head'](block4)        # 151
        return [pool1, pool2, pool3, block4, out]
        
        
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)       
    