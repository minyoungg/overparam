import torch
import torch.nn as nn
import torch.nn.functional as F


def conv2d_identity(identity, inputs):
    """
    Reshapes and pads identity to match input features
    """
    in_sz = identity.size(2)
    scale = round(in_sz / inputs.size(2))

    p = (identity.size(2) - (scale * inputs.size(2))) // 2
    if p > 0:
        identity = identity[:, :, p:in_sz-p, p:in_sz-p]

    identity = F.interpolate(identity, scale_factor=1/scale, mode='nearest')
    identity = pad_to(identity, inputs)
    return identity


def pad_to(src_T, dst_T):
    """ pads src_T to dst_T shape """
    pad_l = (dst_T.size(3) - src_T.size(3)) // 2
    pad_r =  (dst_T.size(3) - src_T.size(3)) - pad_l

    pad_t = (dst_T.size(2) - src_T.size(2)) // 2
    pad_b = (dst_T.size(2) - src_T.size(2)) - pad_t

    return F.pad(src_T, [pad_l, pad_r, pad_t, pad_b], "constant", 0)


def pad(T, p):
    """ pads src_T to dst_T shape """
    return F.pad(T, [p] * 4, "constant", 0)
