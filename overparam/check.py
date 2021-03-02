import warnings
from collections import OrderedDict


def check_arguments(in_dim, out_dim, **kwargs):
    """ Checks if the arguments are valid """
    
    assert kwargs['residual_mode'] in ['learn', 'none']

    if type(kwargs['residual_intervals']) is int:
        kwargs['residual_intervals'] = [kwargs['residual_intervals']]

    if kwargs['residual']:
        for n in kwargs['residual_intervals']:
            if (kwargs['depth'] % n != 0):
                warnings.warn('Depth is not modulo residual_n, the resulting ' + \
                              'blocks may not be of the same size ' +\
                              f'{kwargs["depth"]} mod {n}')

    if kwargs['residual'] and not kwargs['residual_mode'] == 'learn':
        assert in_dim == out_dim,\
            f'expected in_dim == out_dim but found {in_dim} vs {out_dim}'

        if len(kwargs['residual_intervals']) == 1 and \
              (kwargs['residual_intervals'][0] == kwargs['depth'] or \
               kwargs['residual_intervals'][0] == -1):
            pass
        else:
            assert kwargs['width'] == 1, \
                'intermediate width expected to be 1 to ensure consistency'

    if not kwargs['bias'] and kwargs['batch_norm']:
        warnings.warn('Bias is set to False but using batch_norm will ' + \
                      'introduce a bias term')
    return
