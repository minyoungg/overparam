import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
import numpy as np
import warnings

from .check import check_arguments
from .overparam_base import OverparamBaseLayer, construct_computation_graph


class OverparamConv2d(OverparamBaseLayer):
    """
    Expanded conv2d layer with various expansion methods. The expressive power
    of this layer is preserved and can be collapsed into a single linear layer.
    This layer has an explicit collapsing function which is called when eval()
    is called. The batch-norm structure follows that of a resblock.

    To get a correct estimation , the convolutional filter is a
    function of hidden kernel size.

    collapsed kernel_size = kernel_size + sum(hidden_kernel_size) - depth + 1

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_sizes (int or list): Size of the convolution kernels. If list,
            it will override the depth and use len(kernel_sizes) as depth.
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of
            the input. Default: 0
        width (int): width_expansion. Default: 1
        depth (int): depth_expansion. Default: 2
        bias (bool): uses bias for linear layers. Default: True
        batch_norm (bool): batch-normalize the linear layer. Default: False
        residual (bool): uses residual connection f(x) + x
        residual_intervals (int, list): interval frequency to add residual
            connections. For example. residual_intervals of `2`
            will have a similar style to resnets. A special variable `-1` is
            used to add a residual connection from start to end
        residual_mode (str): method for how skip connections are deployed
        negative_slope (float): slope of the expected activation function after
            this layer, assumes ReLU by default. Default: 0
        collapse_on_eval (bool): if False, does not collapse the weight
            and uses the expanded forward even at eval(). Default: True

    Examples::
        >>> layer = EPConv2d(32, 32, kernel_sizes=[5, 3, 1], padding=3)
        >>> x = torch.randn(1, 32, 16, 16)
        >>> out1 = net(x) # uses collapsed
        >>> out2 = net(x, override='expand') # uses expanded
        >>> torch.allclose(out1, out2, atol=1e-5)
        True
        >>> print(layer.weight.size())
        torch.Size([8, 3, 5, 5])
        >>> print(layer.bias.size())
        torch.Size([8])
    """

    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 1], stride=1,
                 padding=1, width=1, depth=2, bias=True, batch_norm=False,
                 residual=False, residual_intervals=2, residual_mode='none',
                 negative_slope=0., collapse_on_eval=True):

        super().__init__()
        check_arguments(in_channels, out_channels, **locals())

        # Conv2d params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels = int(out_channels * width)

        self.stride = stride
        self.padding = padding

        if type(kernel_sizes) is int:
            kernel_sizes = [kernel_sizes] + [1] * (depth - 1)
        else:
            depth = len(kernel_sizes)

        # Expansion params
        self.depth = depth
        self.width = width

        # expanded kernel sizes
        self.kernel_sizes = kernel_sizes

        # effective kernel size
        kernel_size = np.sum(kernel_sizes) - depth + 1
        self.kernel_size = (kernel_size, kernel_size)

        self.residual = residual
        self.residual_mode = residual_mode
        self.negative_slope = negative_slope

        self.collapse_on_eval = collapse_on_eval

        ### (1) create variables for expanded weights
        self.layer_dict, self.graph = \
                        self.construct_graph(
                            depth=depth,
                            residual=residual,
                            residual_mode=residual_mode,
                            residual_intervals=residual_intervals,
                            layer_fn=nn.Conv2d,
                            in_dim=in_channels,
                            out_dim=out_channels,
                            hid_dim=hidden_channels,
                            bias=bias,
                            batch_norm=batch_norm,
                            batch_norm_fn=nn.BatchNorm2d,
                            kernel_sizes=kernel_sizes,
                            f_kwargs={'stride': (stride if depth == 1 else 1),
                                      'padding': padding},
                            m_kwargs={'stride': 1, 'padding': 0},
                            l_kwargs={'stride': stride, 'padding': 0},
                            )

        ### (2) create variables for collapsed weights
        if self.collapse_on_eval:
            self.register_buffer('weight', torch.Tensor(
                        out_channels, in_channels, kernel_size, kernel_size))
            self.register_buffer('bias', torch.zeros(out_channels))

        ### (3) initialize + post processing methods
        self.reset_parameter()
        return


    def reset_parameter(self, init_fn=nn.init.kaiming_normal_, b_init='default',
                        mode='fan_out'):
        """ Initialize weights, preserve output variance """
        # default uses res-net initialization
        return super().reset_parameter(init_fn=init_fn, b_init=b_init, mode=mode)


    def collapsed_forward(self, x):
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding)


    @staticmethod
    def compute_same_padding(kernel_sizes):
        """
        automatically computes padding required to maintain spatial dimension
        from kernel sizes
        """
        if type(kernel_sizes) is int:
            kernel_sizes = [kernel_sizes]
        kernel_sizes = np.sum(kernel_sizes) - len(kernel_sizes) + 1
        padding = max(kernel_sizes // 2, 0)
        return padding
