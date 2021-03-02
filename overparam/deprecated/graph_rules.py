import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cardinal_layer import CardinalWrapper



def find_linear_rule(G):
    """
    Find a minimal linear rule from graph G. A minimal linear rule
    occurs when given a node  W1 W2.
    Args:
        G (dict): computation linkage graph
        layer_dict (dict): ModuleDict from graph linear
        weight_graph (dict): dict consisting of intermediate weights
    Returns:
        node (str): the node associated to W1
        next_node (str): the node associated to W2
    """

    for node, edges in G.items():
        if node in ['s', 't']:
            continue

        if edges['batch_norm']:
            continue

        if len(G[node]['residual_out']) > 0:
            continue


        if len(edges['next']) > 1:
            continue

        next_node = list(edges['next'])[0]

        if next_node in ['s', 't']:
            continue

        if len(G[next_node]['residual_in']) > 0:
            continue

        return node, next_node
    return None


def combine_linear(G, layer_dict, weight_graph, node, next_node):
    """
    Given a minimal linear rule, collapses it into a single weight.
    The update to G is are inplace operations.
    Args:
        G (dict): computation linkage graph
        weight_graph (dict): dict consisting of intermediate computations
        node (str): the node associated to W1
        next_node (str): the node associated to W2
    Returns:
        G (dict): updated computation linkage graph
        weight_graph (dict): updated weight_graph
    """

    replace_node = 't' if next_node == 't' else node
    remove_node = node if next_node == 't' else next_node

    G[replace_node]['prev'] = G[node]['prev']
    G[replace_node]['next'] = G[next_node]['next']
    G[replace_node]['residual_out'] = G[next_node]['residual_out']

    del G[remove_node]
    rename_variable(G, remove_node, replace_node)

    # grab weight
    prev_w, prev_b = get_weight(layer_dict, weight_graph, node)
    curr_w, curr_b = get_weight(layer_dict, weight_graph, next_node)

    if isinstance(layer_dict[node][0], CardinalWrapper):
        layer = layer_dict[node][0].layers[0] # template layer

        # x = torch.randn(1, 32).cuda()
        # y = layer_dict[node][0](x)
        # print(y.mean(), y.std())
        #
        # y = F.linear(x, prev_w, prev_b)
        # print(y.mean(), y.std())
        # print(layer_dict[node][0])
    else:
        layer = layer_dict[node][0]

    ### --- weight combination rule --- ###
    if isinstance(layer, nn.Linear):
        new_w = curr_w @ prev_w

    elif isinstance(layer, nn.Conv2d):
        padding = curr_w.shape[-1] - 1
        new_w = F.conv2d(prev_w.permute(1, 0, 2, 3), curr_w.flip(-1, -2),
                         padding=padding).permute(1, 0, 2, 3)

    else:
        raise ValueError(f'unknown layer {layer_dict[node][0]}')

    ### --- bias combination rule --- ###
    new_b = curr_b
    if prev_b is not None:
        if isinstance(layer, nn.Linear):
            new_b = curr_w @ prev_b

        elif isinstance(layer, nn.Conv2d):
            new_b = curr_w.sum(2).sum(2) @ prev_b

        if curr_b is not None:
            new_b += curr_b

    weight_graph[replace_node] = {new_w, new_b}
    del weight_graph[remove_node]
    return G, weight_graph


######
## Residual rules
######


def find_residual_rule(G):
    """
    Find a minimal residual rule from graph G. A minimal residual rule
    occurs when given a node x W1 y. There is a skip connection from
    x to y.
    Args:
        G (dict): computation linkage graph
    Returns:
        in_node (str): the node associated to W1
        node (str): the node before W1
        next_node (str): the node in which the output we skip from
    """

    for node, edges in G.items():

        if node in ['s', 't']:
            continue

        if edges['batch_norm']:
            continue

        prev_node = list(edges['prev'])[0]
        next_node = list(edges['next'])[0]

        if next_node not in G[prev_node]['residual_out']:
            continue

        return prev_node, node, next_node
    return None


def combine_residual(G, layer_dict, weight_graph, prev_node, node, next_node,
                     normalize_residuals=False):
    """
    Given a minimal residual connection, collapses it into a single weight.
    The update to G is are inplace operations.
    Args:
        G (dict): computation linkage graph
        weight_graph (dict): dict consisting of intermediate computations
        in_node (str): the node associated to W1
        node (str): the node before W1
        next_node (str): the node in which the output we skip from
    Returns:
        G (dict): updated computation linkage graph
        weight_graph (dict): updated weight_graph
    """

    G[next_node]['residual_in'].remove(prev_node)
    G[prev_node]['residual_out'].remove(next_node)

    prev_w, prev_b = get_weight(layer_dict, weight_graph, node)

    if isinstance(layer_dict[node][0], CardinalWrapper):
        layer = layer_dict[node][0].layers[0] # template layer
        num_inputs = layer_dict[node][0].cardinality + 1
    else:
        layer = layer_dict[node][0]
        num_inputs = 2

    if isinstance(layer, nn.Linear):
        new_w = prev_w + torch.eye(prev_w.size(1)).to(prev_w)
    elif isinstance(layer, nn.Conv2d):
        new_w = prev_w + nn.init.dirac_(torch.empty_like(prev_w.clone()))
    else:
        raise RunTimeError(f'unknown layer type {type(layer_dict[node][0])}')

    if normalize_residuals:
        new_w = new_w / math.sqrt(num_inputs)
        if prev_b is not None:
            prev_b = prev_b / math.sqrt(num_inputs)

    weight_graph[node] = {new_w, prev_b}
    return G, weight_graph


######
## Batch-norm rules
######


def find_batch_norm_rule(G):
    """
    Find a batch-norm rule from graph G. A batch-norm rule is W1 -> BN.
    Args:
        G (dict): computation linkage graph
        layer_dict (dict): ModuleDict from graph linear
        weight_graph (dict): dict consisting of intermediate weights
    Returns:
        node (str): the node associated to W1
    """

    for node, edges in G.items():
        if not G[node]['batch_norm']:
            continue

        return node

    return None


def combine_batch_norm(G, layer_dict, weight_graph, node):
    """
    Given a batch norm rule, collapses it into a single weight.
    The update to G is are inplace operations.
    Args:
        G (dict): computation linkage graph
        layer_dict (dict): ModuleDict from graph linear
        weight_graph (dict): dict consisting of intermediate weights
        node (str): the node associated to W1
    Returns:
        G (dict): updated computation linkage graph
        weight_graph (dict): updated weight_graph
    """

    linear_layer, bn_layer = layer_dict[node]
    assert isinstance(bn_layer, (nn.BatchNorm1d, nn.BatchNorm2d))

    w, b = get_weight(layer_dict, weight_graph, node)
    mu, var = bn_layer.running_mean, bn_layer.running_var

    if bn_layer.affine:
        bn_w, bn_b = bn_layer.weight, bn_layer.bias

    if b is None or b is -1:
        b = torch.zeros(w.size(0)).to(w)

    if isinstance(bn_layer, nn.BatchNorm1d):
        new_w = w / torch.sqrt(var.unsqueeze(1) + 1e-5)
        new_b = (b - mu) / torch.sqrt(var + 1e-5)

        if bn_layer.affine:
            new_w = bn_w.unsqueeze(1) * new_w
            new_b = bn_w * new_b + bn_b

    elif isinstance(bn_layer, nn.BatchNorm2d):
        new_w = w / torch.sqrt(var.view(-1, 1, 1, 1) + 1e-5)
        new_b = (b - mu) / torch.sqrt(var + 1e-5)

        if bn_layer.affine:
            new_w = bn_w.view(-1, 1, 1, 1) * new_w
            new_b = bn_w * new_b + bn_b

    G[node]['batch_norm'] = False
    weight_graph[node] = {new_w, new_b}
    return G, weight_graph


######
## miscellaneous helper functions
######


def is_minimal_graph(G):
    """ Checks if the graph is minimal. The exit condition """
    node_cond = (len(G) == 3)
    res_cond = np.all([len(G[k]['residual_in']) == 0 for k in G])
    bn_cond = np.all([G[k]['batch_norm'] == False for k in G])
    return (node_cond and res_cond and bn_cond)


def rename_variable(G, from_node, to_node):
    """ Renames variable in G `from_node` to `to_node` """
    for node, edges in G.items():

        for n in ['prev', 'next', 'residual_in', 'residual_out']:
            connected_nodes = edges[n]

            if connected_nodes is None:
                continue

            for x in connected_nodes:
                if x == from_node:
                    connected_nodes.remove(from_node)
                    connected_nodes.add(to_node)
    return G


def construct_weight_graph(graph):
    """
    Constructs a weight-graph. A weight graph are used to store
    intermediate weights and biases from collaping layers. The weight
    graph is a dictionary consisting of a tuple (weight, bias) where the
    key is the name of the node. Weight and bias are initially set to -1.
    Returns
        weight_graph (dict): dict consisting of intermediate computations
    """
    # weights and bias is added only when needed
    weight_graph = {k: (-1, -1) for k in graph.keys()}
    return weight_graph


def get_weight(layer_dict, weight_graph, node):
    """
    Returns weight from the weight graph. If the weight and bias is empty
    returns the original weight from the layer_dict.
    Args:
        layer_dict (dict): ModuleDict from graph linear
        weight_graph (dict): dict consisting of intermediate weights
        node (str): name of the node to query from
    Returns:
        weight (Tensor): weight of the node
        bias (Tensor or None): bias of the node
    """
    w, b = weight_graph[node]

    if w is -1:
        w = layer_dict[node][0].weight

    if b is -1:
        b = layer_dict[node][0].bias

    return w, b
