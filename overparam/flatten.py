import torch.nn as nn


class Flatten(nn.Module):
    """
    Flatten convolution features into fully-connected features
    Tensor BCHW (BC11) -> BC
    """

    def __init__(self, *args):
        super().__init__()
        return


    def forward(self, x):
        assert x.size(2) == 1 and x.size(3) == 1, \
                f'expected spatial dimension to be 1 x 1 ' +\
                f'but found {x.size(2)} x {x.size(3)}'
        return x.squeeze(2).squeeze(2)
