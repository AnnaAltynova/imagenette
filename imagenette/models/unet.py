import torch 
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Sequential):
    """
    contracting, embedding: in: (c, h, w), out: (c * 2, h, w)
    expansion: in: (c, h, w), out: (c // 2, h, w)
    """
    def __init__(self, in_channels=3, mode='contracting'):
        assert mode in ['contracting', 'expansion', 'embedding']
        blocks = []
        if mode == 'expansion':
            out_channels = in_channels // 2
        else:
            out_channels = in_channels * 2
        if in_channels == 3:
            out_channels = 64
            
        for i in range(2):
            blocks.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=3, padding=1, bias=False))
            blocks.append(nn.BatchNorm2d(num_features=out_channels))
            blocks.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        
        super().__init__(*blocks) 
        
        
class ExpansionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, # BN?
                                           kernel_size=2, stride=2)
        self.convblock = ConvBlock(in_channels=in_channels, mode='expansion')
        
        
    def forward(self, x, emb):
        x = self.upsample(x)
        x = torch.cat([x, emb], dim=1)
        x = self.convblock(x)
        return x


class Unet(nn.Module):
    def __init__(self, input_shape=256, num_classes=151):
        super().__init__()
        in_channels = input_shape[1]
        init_shape = input_shape[2]
        encoder_out_shapes = []
        for i in range(4):
            encoder_out_shapes.append(init_shape)
            init_shape = init_shape // 2
            
        self.encoder = nn.ModuleList([ConvBlock(in_channels=in_channels, mode='contracting'), 
                                      nn.MaxPool2d(kernel_size=2, stride=2)])
        in_channels = 64
        for i in range(3):
            self.encoder.append(ConvBlock(in_channels=in_channels, mode='contracting')) 
            self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels *= 2
        
        self.embedding = ConvBlock(in_channels=in_channels, mode='embedding')
        in_channels *= 2
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(ExpansionBlock(in_channels=in_channels))
            in_channels //= 2
        self.head = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=num_classes)
        
    def forward(self, x):
        enc1 = self.encoder[0](x)
        x = self.encoder[1](enc1)
        enc2 = self.encoder[2](x)
        x = self.encoder[3](enc2)
        enc3 = self.encoder[4](x)
        x = self.encoder[5](enc3)
        enc4 = self.encoder[6](x)
        x = self.encoder[7](enc4)
        
        x = self.embedding(x)
        
        x = self.decoder[0](x, enc4)
        x = self.decoder[1](x, enc3)
        x = self.decoder[2](x, enc2)
        x = self.decoder[3](x, enc1)
        x = self.head(x)
        return x
    
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)  