import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
import sympy as sp


def minimize_linear_rule(computation_graph, layer_dict):
    """
    minimizes stacked linear connections of the form `x -- y -- z`.
    """

    total_count = 0

    while True:
        count = 0

        graph = computation_graph[1]
        vertices = list(graph.keys())

        # (1a) find nodes that does not have incoming residual connection
        check_in = np.array([((len(graph[v]['res']) == 0) and \
                              (not graph[v]['norm'])) for v in vertices[:-1]])
        check_in = np.squeeze(np.argwhere(check_in == True), 1)

        if len(check_in) == 0:
            break

        # (1b) find nodes that does not have outgoing residual connection
        check_out = []
        for idx in check_in:
            node = vertices[idx]
            check_out += [np.all([[node not in graph[v]['res']] \
                            for v in vertices[idx + 1:]])]
        check_out = np.squeeze(np.argwhere(np.array(check_out) == True), 1)

        if len(check_out) == 0:
            break

        idx = check_in[check_out[0]]
        node1, node2 = vertices[idx], vertices[idx + 1]
        edge1, edge2 = graph[node1], graph[node2]

        layer1 = layer_dict[edge1['layer']][0]
        layer2 = layer_dict[edge2['layer']][0]

        if isinstance(layer1, nn.Linear):
            weight = layer2.weight @ layer1.weight

        elif isinstance(layer1, nn.Conv2d):
            padding = layer2.weight.shape[-1] - 1
            weight = F.conv2d(layer1.weight.permute(1, 0, 2, 3),
                              layer2.weight.flip(-1, -2),
                              padding=padding).permute(1, 0, 2, 3)

        else:
            raise ValueError(f'unknown layer {type(layer)}')

        if layer1.bias is not None:
            if isinstance(layer1, nn.Linear):
                bias = layer2.weight @ layer1.bias

            elif isinstance(layer1, nn.Conv2d):
                bias = layer2.weight.sum(2).sum(2) @ layer1.bias

            if layer2.bias is not None:
                bias += layer2.bias
        else:
            bias = None

        # (2) update graph
        edge2['eqn'] = edge2['eqn'].subs(sp.MatrixSymbol(edge2['in'], 1, 1),
                                         sp.MatrixSymbol(edge1['in'], 1, 1))
        edge2['in'] = edge1['in']
        del computation_graph[1][node1]
        computation_graph[0].remove(node1)
        layer_dict[edge2['layer']][0].weight.data = weight

        if isinstance(layer1, nn.Conv2d):
            layer_dict[edge2['layer']][0].padding = \
                        (layer1.padding[0] + layer2.padding[0],
                         layer1.padding[0] + layer2.padding[1])

        if bias is not None:
            layer_dict[edge2['layer']][0].bias.data = bias

        del layer_dict[edge1['layer']]
        
        total_count += 1

    return total_count



def minimize_residual_rule(computation_graph, layer_dict):
    """
    minimizes residual connections of the form `x -- y` where there exists a
    residual connection between `x` and `y`.
    """

    total_count = 0

    while True:
        count = 0

        for edge in computation_graph[1].values():

            if len(edge['res']) == 0:
                continue

            check_res = np.array([(r == edge['in'] and not edge['norm']) \
                                    for r in edge['res']])

            if not np.any(check_res):
                continue

            res_var = edge['res'][np.squeeze(np.argwhere(check_res == True))]

            layer = layer_dict[edge['layer']][0]
            weight = layer.weight
            bias = layer.bias

            if isinstance(layer, nn.Linear):
                weight += torch.eye(weight.size(1)).to(weight)

            elif isinstance(layer, nn.Conv2d):
                weight += nn.init.dirac_(torch.empty_like(weight.clone()))

            else:
                raise RunTimeError(f'unknown layer type {type(layer)}')

            edge['res'].remove(res_var)
            edge['eqn'] -= sp.MatrixSymbol(res_var, 1, 1)
            layer_dict[edge['layer']][0].weight = weight

            count += 1
            total_count += 1

        if count == 0:
            break

    return total_count


def minimize_batch_norm_rule(computation_graph, layer_dict):
    """
    minimizes batch norm of the form `x -- y -- bn`
    Warning: batch-norm introduces bias even when it was originally set to False
    """

    total_count = 0

    while True:
        count = 0

        for edge in computation_graph[1].values():

            if not edge['norm']:
                continue

            layer = layer_dict[edge['layer']][0]
            bn = layer_dict[edge['layer']][1]

            if layer.bias is None:
                _bias = torch.zeros(layer.weight.size(0)).to(layer.weight)
            else:
                _bias = layer.bias

            if isinstance(bn, nn.BatchNorm1d):
                weight = layer.weight / torch.sqrt(bn.running_var.unsqueeze(1) + 1e-5)
                bias = (_bias - bn.running_mean) / torch.sqrt(bn.running_var + 1e-5)

                if bn.affine:
                    weight = bn.weight.unsqueeze(1) * weight
                    bias = bn.weight * bias + bn.bias

            elif isinstance(bn, nn.BatchNorm2d):
                weight = layer.weight / torch.sqrt(bn.running_var.view(-1, 1, 1, 1) + 1e-5)
                bias = (_bias - bn.running_mean) / torch.sqrt(bn.running_var + 1e-5)

                if bn.affine:
                    weight = bn.weight.view(-1, 1, 1, 1) * weight
                    bias = bn.weight * bias + bn.bias

            else:
                raise RunTimeError(f'unknown layer type {type(bn)}')

            layer_dict[edge['layer']][0].weight.data = weight
            layer_dict[edge['layer']][0].bias = Parameter(bias)

            edge['norm'] = False
            count += 1
            total_count += 1

        if count == 0:
            break

    return total_count
