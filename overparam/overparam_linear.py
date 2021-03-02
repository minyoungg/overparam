import os
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from .check import check_arguments
from .overparam_base import OverparamBaseLayer, construct_computation_graph


class OverparamLinear(OverparamBaseLayer):
    """
    Expanded linear layer with various expansion methods. The expressive power
    of this layer is preserved and can be collapsed into a single linear layer.
    This layer has an explicit collapsing function which is called when eval()
    is called. The batch-norm structure follows that of a resblock.

    Args:
        in_features (int): size of each input sample
        out_features (int): size of each output sample
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
        >>> layer = EPLinear(32, 32, depth=2, batch_norm=True)
        >>> x = torch.randn(16, 32)
        >>> net(x) # warm up for batch-norm
        >>> net.eval() # collapses and defaults the forward pass to use it
        >>> out1 = net(x) # uses collapsed
        >>> out2 = net(x, override='expand') # uses expanded
        >>> torch.allclose(out1, out2, atol=1e-5)
        True
        >>> print(layer.weight.size())
        torch.Size([64, 32])
        >>> print(layer.bias.size())
        torch.Size([64])
    """

    def __init__(self, in_features, out_features, width=1, depth=2, bias=True,
                 batch_norm=False, residual=False, residual_intervals=2,
                 residual_mode='none', negative_slope=0., collapse_on_eval=True):

        super().__init__()
        check_arguments(in_features, out_features, **locals())

        # Linear params
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features = int(out_features * width)

        # Expansion params
        self.depth = depth
        self.width = width

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
                                    layer_fn=nn.Linear,
                                    in_dim=in_features,
                                    out_dim=out_features,
                                    hid_dim=hidden_features,
                                    bias=bias,
                                    batch_norm=batch_norm,
                                    batch_norm_fn=nn.BatchNorm1d,
                                    )

        ### (2) create variables for collapsed weights
        if self.collapse_on_eval:
            self.register_buffer('weight', torch.Tensor(out_features, in_features))
            self.register_buffer('bias', torch.zeros(out_features))

        ### (3) initialize + post processing methods
        self.reset_parameter()
        return


    def reset_parameter(self, init_fn=nn.init.kaiming_normal_, b_init='default',
                        mode='fan_in'):
        """ Initialize weights, preserve output variance """
        return super().reset_parameter(init_fn=init_fn, b_init=b_init, mode=mode)


    def collapsed_forward(self, x):
        """ Forward with collapsed weight """
        return F.linear(x, self.weight, self.bias)
