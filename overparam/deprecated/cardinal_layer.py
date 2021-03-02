import math

import torch
import torch.nn as nn

from .conv_helpers import conv2d_identity


class CardinalWrapper(nn.Module):
    """
    Different from ResNext group conv implementation.
    Same input passed throught each branch.
    """
    def __init__(self, layer, cardinality, in_dim, out_dim, residual, *args, **kwargs):
        super().__init__()
        layers = []

        for _ in range(cardinality):
            layers += [layer(in_dim, out_dim, *args, **kwargs)]

        self.layers = nn.ModuleList(layers)
        self.cardinality = cardinality
        self.residual = residual
        return


    def forward(self, x):
        feats = []

        for f in self.layers:
            feats += [f(x)]

        feats = torch.stack(feats)
        feats = feats.sum(0)

        if self.residual:
            if isinstance(self.layers[0], nn.Linear):
                feats = (x + feats) / math.sqrt(self.cardinality + 1)
            else:
                feats = (conv2d_identity(x, feats) + feats) / math.sqrt(self.cardinality + 1)
        else:
            feats =  feats /  math.sqrt(self.cardinality)

        return feats


    def reset_parameter(self, weight_init_fn, bias_init_fn):
        for layer in self.layers:
            weight_init_fn(layer)
            bias_init_fn(layer)
        return


    @property
    def weight(self):
        w = torch.stack([w.weight.data for w in self.layers]).sum(0)
        if self.residual:
            if isinstance(self.layers[0], nn.Linear):
                w += torch.eye(w.size(1)).to(w)
            elif isinstance(self.layers[0], nn.Conv2d):
                print('here')
                w += nn.init.dirac_(torch.empty_like(w.clone()))
            return w / math.sqrt(self.cardinality + 1)
        else:
            return w / math.sqrt(self.cardinality)


    @property
    def bias(self):
        if self.layers[0].bias is None:
            return None
        b = torch.stack([w.bias.data for w in self.layers]).sum(0)
        if self.residual:
            return b / math.sqrt(self.cardinality + 1)
        else:
            return b / math.sqrt(self.cardinality)
