# PyTorch linear over-parameterized layers

## 1. Installation
<b> Developed on </b> 
- <b>Python 3.7 </b> :snake:
- <b>PyTorch 1.7</b> :fire:

```bash
> git clone https://github.com/minyoungg/overparam-minimal
> cd overparam-minimal
> pip install .
```

## 2. Usage
Currently supports overparameterized linear (`EPLinear`) and conv2d layers (`EPConv2d`).   
`EP` is just an abbreviation for expand-and-project. 

The layers work exactly the same as any `torch.nn` layers. Few important arguments to take notice of are:
`depth`, `width`, `residual`, `batch_norm`. Refer to the files for the full documentation.

<br>

###  EPLinear layer (equivalence: `nn.Linear`) 

```python
from overparam_layers import EPLinear
 
layer = EPLinear(4, 8, width=1, depth=4)
x = torch.randn(1, 4)

# Forward pass (expanded form)
layer.train()
y1 = layer(x)

# Forward pass (collapsed form) [automatic]
layer.eval()
y2 = layer(x)
```

To access the collapsed weights
```python
layer.eval()
print(layer.weight)
print(layer.bias)
```

<b> Supported parameterization </b>  
Let `x` be the input, `F` and `G` be a set of linear layers, and `y` be the output.

> Residual connection
``` y = x + F(x) ```

> Learnable residual connection
``` y = G(x) + F(x) ```

> Batch normalization (can be combined with residual connection)
``` y = x + BN1D(F(x)) ```

<br>



### EPConv2d layer (equivalence: `nn.Conv2d`) 

<b>(note)</b>: kernel_size is a function of the individual kernels

To create a 3 layer of Convolution2D with `5x5`, `3x3`, `1x1` kernels.
```python
from overparam_layers import EPConv2d
import numpy as np

kernel_sizes = [5, 3, 1]
```

Compute the required padding to maintain spatial dimension
```python
padding = max((np.sum(kernel_sizes) - len(kernel_sizes) + 1) // 2, 0)
```

The effective kernel size is 
```
print(max((np.sum(kernel_sizes) - len(kernel_sizes) + 1) // 2, 0))
```

Now we can apply forward pass as usual
```python
layer = EPConv2d(2, 4, kernel_sizes, padding, depth=len(kernel_sizes))
x = torch.randn(1, 2, 8, 8)

# Forward pass (expanded form)
layer.train()
y1 = layer(x)

# Forward pass (collapsed form) [automatic]
layer.eval()
y2 = layer(x)
```

To access the collapsed weights
```python
layer.eval()

print(layer.weight)
print(layer.bias)
```



<b> Supported parameterization </b>  
Let `x` be the input, `F` and `G` be a set of linear layers, and `y` be the output.

> Residual connection
``` y = x + F(x) ```

> Learnable residual connection
``` y = G(x) + F(x) ```

> Batch normalization (can be combined with residual connection)
``` y = x + BN2D(F(x)) ```


