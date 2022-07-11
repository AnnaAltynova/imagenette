import torch

def pixel_shuffle(inputs, upscale_factor):
    """Rearranges elements in a tensor of shape [*, C*r^N, D1, D2, ... DN] to a
    tensor of shape [*, C, D1*r, D2*r, ... DN*r].
    
    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.
    
    Args:
        inputs (Variable): Inputs
        upscale_factor (int): factor to increase spatial resolution by
    Examples:
        >>> inputs = torch.randn(1, 64, 10, 10, 10))
        >>> output = pixel_shuffle(inputs, 4)
        >>> print(output.size())
        torch.Size([1, 1, 40, 40, 40])
    """
    batch_size, channels, *dims = inputs.size()
    N = len(dims)
    channels //= (upscale_factor ** N)
    inputs_view = inputs.contiguous().view(batch_size, channels, *(upscale_factor for i in range(N)), *dims) # (bs, ch, r, ... r, d1, .... dN)

    permute = [ None for i in range(2 * N)]
    permute[::2] = range(N + 2, 2 * N + 2)
    permute[1::2] = range(2, N + 2)
    
    # (bs, ch, r, ... r, d1, .... dN) -> (bs, ch, d1, r, d2, r, ... dN, r), idx: (0, 1, N + 2, 2, ... N + 1 + i, 2 + i, ... 2N + 1, N + 1)
    shuffle_out = inputs_view.permute(0, 1, *permute).contiguous()  
    out_dims = [dim * upscale_factor for dim in dims]
    return shuffle_out.view(batch_size, channels, *out_dims)


def pixel_unshuffle(inputs, downscale_factor):
    """Reverses `pixel_shuffle` operation by rearranging elements
    in a tensor of shape [*, C, D1*r, D2*r, ... DN*r] to a tensor of shape
    [*, C*r^N, D1, DN], where r is a downscale factor.

    See the paper:
    `Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network`_
    by Shi et. al (2016) for more details.
    
    Args:
        inputs (Variable): Inputs
        downscale_factor (int): factor to decrease spatial resolution by
    """
    batch_size, channels, *dims = inputs.size()
    N = len(dims)

    out_channels = channels * downscale_factor ** N
    out_dims = [dim // downscale_factor for dim in dims]

    reshape = [ None for i in range(2 * N)]
    reshape[::2] = out_dims
    reshape[1::2] = [downscale_factor for i in range(N)]

    after_view = inputs.view(batch_size, channels, *reshape)    # undo the last view

    # (bs, ch, d1, r, d2, r, ... dN, r) -> (bs, ch, r, ... r, d1, .... dN) , idx: (0, 1, 3, 5, ... 2N + 1, 2, 4, ... 2N)
    permute = [ None for i in range(2 * N)]
    permute[:2 * N] = range(3, 2 * N + 2, 2)
    permute[2 * N:] = range(2, 2 * N + 1, 2)

    after_shuffle = after_view.permute(0, 1, *permute).contiguous()    # undo permutation
    return after_shuffle.view(batch_size, out_channels, *out_dims).contiguous()    # undo the first view