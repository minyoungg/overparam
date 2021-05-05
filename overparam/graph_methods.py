import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import copy
import numpy as np
import sympy as sp

from .conv_helpers import conv2d_identity, pad_to
from .graph_minimizers import minimize_linear_rule, \
                              minimize_residual_rule, \
                              minimize_batch_norm_rule


def graph_forward(inputs, computation_graph, layer_dict, return_intermediate=False):
    """ forward pass using computation graph """
    # intermediate features for graph computation
    verticies, edges = computation_graph
    intermediate_computations = {'s': inputs}

    for v in verticies[1:]:
        inputs = intermediate_computations[str(edges[v]['in'])]

        layer = layer_dict[edges[v]['layer']]
        outputs = layer(inputs)

        residual_inputs = 0
        for rv in edges[v]['res']:
            res_in = intermediate_computations[rv]

            if isinstance(layer[0], nn.Conv2d):
                res_in = conv2d_identity(res_in, outputs)

            residual_inputs = residual_inputs + res_in

        outputs = outputs + residual_inputs
        intermediate_computations[edges[v]['out']] = outputs

    if return_intermediate:
        return intermediate_computations

    return intermediate_computations['t']


@torch.no_grad()
def graph_collapse(computation_graph, layer_dict):
    """ use symbolic representation to expand out the rules """
    device = next(layer_dict.parameters()).device
    layer_dict = copy.deepcopy(layer_dict).cpu()
    computation_graph = copy.deepcopy(computation_graph)
    verticies, edges = computation_graph

    # (1) minimize graph as far as you can
    while True:
        found_lin = minimize_linear_rule(computation_graph, layer_dict)
        found_res = minimize_residual_rule(computation_graph, layer_dict)
        found_bn = minimize_batch_norm_rule(computation_graph, layer_dict)

        if (found_lin + found_res + found_bn) == 0:
            break

    # (2) simplify non-trivial connections
    for i, v in enumerate(verticies[1:][::-1]):
        if i == 0:
            eqn = computation_graph[1][v]['eqn']
        else:
            sub_eqn = computation_graph[1][v]['eqn']
            out_var = sp.MatrixSymbol(computation_graph[1][v]['out'], 1, 1)
            eqn = eqn.subs(out_var, sub_eqn)

    w_eqn, b_eqn = parse_equation(eqn.expand())

    weight = equation_to_weight(w_eqn, layer_dict)
    bias = equation_to_bias(b_eqn, layer_dict)

    if bias is not None:
        bias = bias.data.to(device)

    return weight.data.to(device), bias


def parse_equation(eqn):
    eqn_str = str(eqn.expand()).split('+')
    w_eqn, b_eqn = [], []
    for eqn_chunk in eqn_str:
        eqn_chunk = eqn_chunk.strip()

        if 's' == eqn_chunk:
            w_eqn.append('I')
        elif 's' in eqn_chunk:
            w_eqn.append(eqn_chunk)
        else:
            b_eqn.append(eqn_chunk)

    w_eqn = '+'.join([x.replace('*s', '') for x in w_eqn])
    b_eqn = '+'.join(b_eqn)
    return w_eqn, b_eqn


@torch.no_grad()
def equation_to_weight(w_eqn, layer_dict):
    weights = []

    _layer = list(layer_dict.values())[0][0] # sample layer

    for w_part in w_eqn.split('+'):
        weight_part = None

        for w_str in w_part.split('*')[::-1]:
            idx = w_str[1:]

            # (1) query weight / bias
            if w_str[0] == 'W':
                m = layer_dict[idx][0].weight

            elif w_str[0] == 'I':

                if isinstance(_layer, nn.Linear):
                    m = torch.eye(weights[0].size(1)).to(weights[0])

                elif isinstance(_layer, nn.Conv2d):
                    m = nn.init.dirac_(torch.empty_like(weight.clone()))

                else:
                    raise RunTimeError(f'unknown layer type {type(layer)}')

            elif w_str[0] == 'b':
                m = layer_dict[idx][0].bias

            else:
                raise RunTimeError(f'unknown equation string {w_str}')

            # (2) matrix multiply
            if weight_part is None:
                weight_part = m

            else:

                if isinstance(_layer, nn.Linear):
                    weight_part = m @ weight_part

                elif isinstance(_layer, nn.Conv2d):
                    padding = m.shape[-1] - 1
                    weight_part = F.conv2d(weight_part.permute(1, 0, 2, 3),
                                           m.flip(-1, -2),
                                           padding=padding).permute(1, 0, 2, 3)

                else:
                    raise RunTimeError(f'unknown layer type {type(layer)}')

        weights.append(weight_part)


    if isinstance(_layer, nn.Linear):
        weight = torch.stack(weights).sum(0)

    elif isinstance(_layer, nn.Conv2d):
        max_size = np.max([w.size(2) for w in weights])
        max_idx = np.squeeze(np.argmax([w.size(2) for w in weights]))

        weight = 0.
        for w in weights:
            weight += pad_to(w, weights[max_idx])

    return weight


@torch.no_grad()
def equation_to_bias(b_eqn, layer_dict):
    biases = None

    _layer = list(layer_dict.values())[0][0]

    for b_part in b_eqn.split('+'):
        bias_part = None

        for b_str in b_part.split('*'):
            idx = b_str[1:]

            # (1) query weight / bias
            if b_str[0] == 'W':
                m = layer_dict[idx][0].weight

                if isinstance(_layer, nn.Conv2d):
                    m = m.sum(2).sum(2)
            else:
                m = layer_dict[idx][0].bias

            # (2) matrix multiply
            if m is None:
                bias_part = None
                break

            if bias_part is None:
                bias_part = m

            elif m is not None:
                bias_part = bias_part @ m

        if bias_part is not None:
            if biases is None:
                biases = bias_part
            else:
                biases += bias_part
    return biases
