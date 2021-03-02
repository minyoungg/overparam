import os, sys

def suppress_print(suppress):
    # add trace on error?
    def suppress_function(func):
        def func_wrapper(*args, **kwargs):
            if suppress:
                sys.stdout = open(os.devnull, 'w')
                value = func(*args, **kwargs)
                sys.stdout = sys.__stdout__
            else:
                value = func(*args, **kwargs)
            return value
        return func_wrapper
    return suppress_function
