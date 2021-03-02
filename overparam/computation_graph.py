import torch
import torch.nn as nn
import torch.nn.init as init

import sympy as sp

import math
import warnings
from collections import OrderedDict



def construct_computation_graph(
                depth, residual, residual_intervals, layer_fn,
                in_dim, out_dim, hid_dim, bias, batch_norm,
                batch_norm_fn, f_kwargs={}, m_kwargs={}, l_kwargs={},
                kernel_sizes=None, residual_mode='none'):
    """
    Constructs a computation graph given the provided configuration.
    Node represents the intermediate features.
    Edges correspond to the functional operator.

    Args:
        see `OverparamLinear` or `OverparamConv2d`
    Returns:
        (V, E): a tuple consisting of ordered `V` verticies. `E` is a dict
            where the keys are the output nodes. `E['eqn']` is the symbolic
            rule that determines the graph. `E['in_sym']` and `E['out_sym']`
            are the sympy symbolic variables.
            `E['in']` and `E['out']` and `E['res']` are string representation
            that defines the input output relationships. `E['res']` is a list.
        graph_dict: torch ModuleDict that consits of the layer parameters
    """

    if type(residual_intervals) is int:
        residual_intervals = [residual_intervals]

    residual_intervals = [x if x != -1 else depth for x in residual_intervals]
    residual_intervals = list(OrderedDict.fromkeys(residual_intervals))

    if residual_mode == 'learn':
        raise NotImplementedError('learnable residual connection not implemented.')

    variables = ['s', *[f'x{i+1}' for i in range(depth - 1)], 't']
    computation_graph = [variables, {}]
    layer_dict = {}

    for i in range(depth):

        ### --- (1) arguments --- ###
        if kernel_sizes is not None:
            k_sz = kernel_sizes[i]
            for kwargs in [f_kwargs, m_kwargs, l_kwargs]:
                kwargs.update({'kernel_size': k_sz})

        # [first layer]
        if i == 0:
            kwargs = f_kwargs
            _in_dim, _out_dim = in_dim, out_dim if depth == 1 else hid_dim

        # [last layer]
        elif i == depth - 1:
            kwargs = l_kwargs
            _in_dim, _out_dim = hid_dim, out_dim

        # [intermediate layers]
        else:
            kwargs = m_kwargs
            _in_dim, _out_dim = hid_dim, hid_dim


        ### --- (2) make layer --- ###
        layer = [layer_fn(_in_dim, _out_dim, bias=bias, **kwargs)]

        if batch_norm:
            num_channels = hid_dim if i < depth - 1 else out_dim
            layer += [batch_norm_fn(num_channels, affine=True)]

        layer_dict[f'{i}'] = nn.Sequential(*layer)


        ### --- (3) symbolic rule --- ###
        in_var = variables[i]
        x = sp.MatrixSymbol(in_var, 1, 1)
        w = sp.MatrixSymbol(f'W{i}', 1, 1)
        b = sp.MatrixSymbol(f'b{i}', 1, 1)
        eqn = w * x + b

        out_var = variables[i + 1]
        y = sp.MatrixSymbol(out_var, 1, 1)

        computation_graph[1][out_var] = {'eqn': eqn, 'in': in_var,
                                         'out': out_var, 'res':[],
                                         'layer': f'{i}', 'norm': batch_norm}

    if residual:

        for n in residual_intervals:
            m = n
            while m <= depth:
                res_var = variables[m - n]
                curr_var = variables[m]

                computation_graph[1][curr_var]['res'].append(res_var)
                computation_graph[1][curr_var]['eqn'] += sp.MatrixSymbol(res_var, 1, 1)
                m += n

    return nn.ModuleDict(layer_dict), computation_graph
