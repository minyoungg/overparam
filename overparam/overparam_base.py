import torch
import torch.nn as nn
import torch.nn.init as init

import sympy as sp

import math
import warnings
from collections import OrderedDict

from .computation_graph import construct_computation_graph
from .graph_methods import graph_forward, graph_collapse



class OverparamBaseLayer(nn.Module):
    """ Template layer for overparam layers """

    def __init__(self):
        super().__init__()
        self.layer_dict = None
        self.graph = None
        self.collapse_on_eval = True

        self.register_buffer('weight', None)
        self.register_buffer('bias', None)
        return


    def construct_graph(self, *args, **kwargs):
        return construct_computation_graph(*args, **kwargs)


    def train(self, mode=True):
        """ Set train mode and implicitely collapse current weight """

        self.training = mode
        for module in self.children():
            module.train(mode)

        if not mode:
            self.weight, self.bias = graph_collapse(self.graph, self.layer_dict)
        return


    def expanded_forward(self, x):
        return graph_forward(x, self.graph, self.layer_dict)


    def collapsed_forward(self, x):
        raise NotImplementedError


    def forward(self, x, override=False):
        if not self.collapse_on_eval:
            warnings.warn('collapse_on_eval set to `False` using expanded forward.')
            return self.expanded_forward(x)

        if self.training or override:
            return self.expanded_forward(x)

        return self.collapsed_forward(x)


    def reset_parameter(self, init_fn=nn.init.kaiming_normal_, b_init='zero',
                        mode='fan_out'):
        """
        Initialize weight

        Args:
            init_fn (torch.nn.init): initialization function
            b_init (str): bias initialization
            mode (str): either fan_in or fan_out
        """

        assert b_init in ['zero', 'default'], f'unknown b_init {b_init}'

        def init_bias(m):

            if m.bias is not None:
                if b_init == 'zero':
                    nn.init.zeros_(m.bias)

                elif b_init == 'default':

                    if isinstance(m, nn.Conv2d):
                        n = m.in_channels
                        for k in m.kernel_size:
                            n *= k
                        stdv = 1. / math.sqrt(n)
                        m.bias.data.uniform_(-stdv, stdv)

                    elif isinstance(m, nn.Linear):
                        fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / math.sqrt(fan_in)
                        nn.init.uniform_(m.bias, -bound, bound)

                    else:
                        raise RunTimeError(f'unexpected layer {type(m)}')
            return


        for i, (node, edges) in enumerate(self.layer_dict.items()):

            # -- (1) compute weight gains -- #

            m = self.layer_dict[node][0]
            assert isinstance(m, (nn.Linear, nn.Conv2d))

            if i == len(self.layer_dict) - 1: # [last layer]
                a = self.negative_slope
            else:
                a = 1. # [intermediate] assumes linear

            init_fn(m.weight, a=a, mode=mode, nonlinearity='leaky_relu')
            init_bias(m)


            # -- (2) batchnorm gain fixing -- #

            if len(self.layer_dict[node]) == 1:
                continue

            m = self.layer_dict[node][1]
            assert isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))

            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)

        return

    def visualize(self, save_path='graph.png'):
        from .vis import visualize_graph
        visualize_graph(self.graph, save_path)
        return
