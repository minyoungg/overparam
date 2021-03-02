import os, sys
import torch
import torch.nn as nn
from .overparam_linear import OverparamLinear
from .overparam_conv import OverparamConv2d


def dev_mode(func, debug=False):
    def func_wrapper(*args, **kwargs):
        sys.stdout = open(os.devnull, 'w')
        value = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return value
    return func_wrapper



def overparameterize(model, depth=2, width=1, overparam='all', residual=False,
                     residual_intervals=2, batch_norm=False, verbose=False,
                     ignore=[], consistent_init=False, collapse_on_eval=True):
    """
    Given a model (e.g. AlexNet and ResNet), replace Linear and Conv2d layer
    with expanded counterparts.

    Args:
        model (torch model): model to replace, existing layers are deletered
        verbose (bool): prints layers being replaced
        ignore (list): Module to ignore in conversion (see example)
        consistent_init (bool): the ignored variables use EP counterpart to use
            same initialization as the rest of the model.
        collapse_on_eval (bool): collapses layers, setting it to False will
            save some memory.
        overparam (bool): if 'all' overparameterize , 'linear', 'conv'
        other-args: refer to EPLinear and EPConv2d

    Examples::
        >>> model = torchvision.models.alexnet()
        # replaces every layer
        >>> overparameterize(model, depth=2)
        # only conv layers
        >>> overparameterize(model, depth=2, ignore=[model.features])

    NOTE: may not always work, so make sure to print the model.
    """
    saved_args = locals()
    assert overparam in ['all', 'conv', 'fc']

    for i, (name, layer) in enumerate(model.named_children()):

        if layer in ignore:
            print(f'ignoring ({name}): {layer}')
            if consistent_init:
                if verbose:
                    print(f'consistent init ({name}): {layer}')

                if isinstance(layer, (nn.Linear, nn.Conv2d)):
                    new_layer = nn.Sequential(layer)
                    setattr(model, name, new_layer)

                _args = {'model':layer, 'depth':1,
                         'residual':False, 'residual_intervals':1,
                         'batch_norm':False, 'verbose':verbose}
                overparameterize(**_args)
            else:
                continue

        else:
            if isinstance(layer, (nn.Linear, nn.Conv2d)):

                if verbose:
                    print(f'replacing ({name}): {layer}')

                if isinstance(layer, nn.Conv2d):
                    if overparam in ['all', 'conv']:
                        new_layer = OverparamConv2d(
                                             layer.in_channels,
                                             layer.out_channels,
                                             kernel_sizes=layer.kernel_size[0],
                                             stride=layer.stride,
                                             padding=layer.padding,
                                             depth=depth,
                                             width=width,
                                             bias=(layer.bias is not None),
                                             batch_norm=batch_norm,
                                             residual=residual,
                                             residual_intervals=residual_intervals,
                                             collapse_on_eval=collapse_on_eval)
                    else:
                        if consistent_init:
                            new_layer = OverparamConv2d(
                                             layer.in_channels,
                                             layer.out_channels,
                                             kernel_sizes=layer.kernel_size[0],
                                             stride=layer.stride,
                                             padding=layer.padding,
                                             depth=1, width=1,
                                             bias=(layer.bias is not None),
                                             collapse_on_eval=collapse_on_eval
                                             )
                        else:
                            continue

                elif isinstance(layer, nn.Linear):
                    if overparam in ['all', 'fc']:
                        new_layer = OverparamLinear(
                                              layer.in_features,
                                              layer.out_features,
                                              depth=depth,
                                              width=width,
                                              bias=(layer.bias is not None),
                                              batch_norm=batch_norm,
                                              residual=residual,
                                              residual_intervals=residual_intervals,
                                              collapse_on_eval=collapse_on_eval
                                              )
                    else:
                        if consistent_init:
                            new_layer = OverparamLinear(
                                              layer.in_features,
                                              layer.out_features,
                                              depth=1,
                                              width=1,
                                              bias=(layer.bias is not None),
                                              collapse_on_eval=collapse_on_eval
                                              )
                        else:
                            continue


                if layer.weight.is_cuda:
                    new_layer = new_layer.cuda()

                setattr(model, name, new_layer)
                del layer

            else:
                saved_args['model'] = layer
                overparameterize(**saved_args)

    return model
