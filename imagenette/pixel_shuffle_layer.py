""" Layer implementing Pixel Shuffle for an arbitrary dimensionality.
Shi et al. "`Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network_
<https://arxiv.org/pdf/1609.05158.pdf>`"
"""
import torch
from torch import nn


class PixelShuffle(nn.Module):
    """ Rearranges elements in a tensor of shape [*, C*r^N, D1, D2, ... DN] to a tensor of shape [*, C, D1*r, D2*r, ... DN*r]
    where r is an upscale factor.
    
    Parameters
    ----------
    upscale_factor : int
        factor to increase spatial resolution by.
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor 
    
    
    def forward(self, x):
        batch_size, channels, *dims = x.size()
        N = len(dims)
        channels //= (self.upscale_factor ** N)
        x = x.contiguous().view(batch_size, channels, *(self.upscale_factor for i in range(N)), *dims) 

        permute = [None for i in range(2 * N)]
        permute[::2] = range(N + 2, 2 * N + 2)
        permute[1::2] = range(2, N + 2)
        x = x.permute(0, 1, *permute).contiguous()  
        
        out_dims = [dim * self.upscale_factor for dim in dims]
        x =  x.view(batch_size, channels, *out_dims)
        return x
    

class PixelUnshuffle(nn.Module):
    
    """ Reverses `PixelShuffle` operation by rearranging elements in a tensor of shape [*, C, D1*r, D2*r, ... DN*r]
    to a tensor of shape [*, C*r^N, D1, DN], where r is a downscale factor.

    Parameters
    ----------
    downscale_factor : int 
        factor to decrease spatial resolution by.
    """
    def __init__(self, downscale_factor):
        super().__init__()
        self.downscale_factor = downscale_factor
        
    def forward(self, x):
        batch_size, channels, *dims = x.size()
        N = len(dims)

        out_dims = [dim // self.downscale_factor for dim in dims]
        reshape = [None for i in range(2 * N)]
        reshape[::2] = out_dims
        reshape[1::2] = [self.downscale_factor for i in range(N)]
        
        # Undo the last view
        x = x.view(batch_size, channels, *reshape)    

        # Undo permutation
        permute = [ None for i in range(2 * N)]
        permute[:2 * N] = range(3, 2 * N + 2, 2)
        permute[2 * N:] = range(2, 2 * N + 1, 2)
        x = x.permute(0, 1, *permute).contiguous()
        
        # Undo the first view
        out_channels = channels * self.downscale_factor ** N
        x = x.view(batch_size, out_channels, *out_dims).contiguous()
        return x
