import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import copy
import math
import numpy as np
import warnings
from pprint import pprint
from collections import OrderedDict

from .graph_rules import (find_linear_rule, combine_linear,
                          find_residual_rule, combine_residual,
                          find_batch_norm_rule, combine_batch_norm,
                          is_minimal_graph, rename_variable,
                          construct_weight_graph, get_weight)
from .conv_helpers import conv2d_identity
from .decorators import suppress_print



class LinearGraphLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_dict = None
        self.graph = None
        self.module_name = None

        self.register_buffer('weight', None)
        self.register_buffer('bias', None)
        return

    @suppress_print(suppress=False)
    def _forward(self, x):

        input_node = 's'
        start_node = '0_0'

        # intermediate features for graph computation
        comps = {input_node: x}
        computation_queue = [start_node]

        while True:
            # ordered set
            computation_queue = list(dict.fromkeys(computation_queue))

            current_node = computation_queue.pop(0)
            E = self.graph[current_node]

            # assume additive relationship of inputs
            assert len(E['prev']) == 1
            inputs = comps[list(E['prev'])[0]]

            res_ins = None
            num_inputs = 1

            if len(E['residual_in']) > 0:
                res_ins = 0

                for r in E['residual_in']:
                    # hacky work around but should be good for now
                    res_in = comps[r]

                    if self.__class__.__name__ == 'EPConv2d':
                        #res_in = conv2d_identity(inputs, res_in)
                        res_in = conv2d_identity(res_in, inputs)

                    res_ins += res_in
                    num_inputs += 1


                # (2) all paths are created equal [default]
                inputs += res_ins


            if current_node == 't':
                assert len(computation_queue) == 0, 'incorrect graph'
                break

            # update if residual
            if len(E['residual_in']) > 0:
                for k in E['prev']:
                    if k in E['residual_in']:
                        continue
                    comps[k] = inputs

            # forward pass and store
            comps[current_node] = self.layer_dict[current_node](inputs)

            # update and store
            computation_queue.extend(list(E['next']))

        return inputs

    @suppress_print(suppress=False)
    @torch.no_grad()
    def _collapse(self):
        """
        WARNING: THIS FUNCTION DOESNT WORK WHEN RESIDUALS ARE INTERLEAVED.

        Collapses the graph into a single linear layer. There is 3 types of
        rules that are supported as of now.
        > (1) Linear - Linear
        > (2) Residual
        > (3) Linear - Batch-Norm
        """

        G = copy.deepcopy(self.graph)
        weight_G = construct_weight_graph(G)

        #pprint(G)
        while not is_minimal_graph(G):

            # (1) batch norm rule
            b = find_batch_norm_rule(G)
            if b is not None:
                # print(f'\n[bn rule] {b}')
                combine_batch_norm(G, self.layer_dict, weight_G, b)
                continue

            # (2) residual rule
            r = find_residual_rule(G)
            if r is not None:
                # print(f'\n[res rule] {r}')
                combine_residual(G, self.layer_dict, weight_G, *r)
                continue

            # (3) residual rule
            l = find_linear_rule(G)
            if l is not None:
                # print(f'\n[lin rule] {l}')
                combine_linear(G, self.layer_dict, weight_G, *l)
                continue

        weight, bias = get_weight(self.layer_dict, weight_G, '0_0')

        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)
        return
