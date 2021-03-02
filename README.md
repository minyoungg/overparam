# PyTorch linear over-parameterized layers

## 1. Installation
<b> Developed on </b> 
- <b>Python 3.7 </b> :snake:
- <b>PyTorch 1.7</b> :fire:

```bash
> git clone https://github.com/minyoungg/overparam
> cd overparam
> pip install .
```

## 2. Usage
The layers work the same as any `torch.nn` layers. Important arguments are `depth`, `width`, `residual`, `batch_norm`. 
Refer to the files for the full documentation.

<br>
###  Getting started

#### (1a) OverparamLinear layer (equivalence: `nn.Linear`) 

```python
from overparam import OverparamLinear
 
layer = OverparamLinear(16, 32, width=1, depth=2)
x = torch.randn(1, 16)
```

#### (1b) OverparamConv2d layer (equivalence: `nn.Conv2d`)

```python
from overparam import OverparamConv2d
import numpy as np
```

We can construct 3 Conv2d layers with kernel dimensions of `5x5`, `3x3`, `1x1`
```python
# Same padding
padding = max((np.sum(kernel_sizes) - len(kernel_sizes) + 1) // 2, 0)

layer = OverparamConv2d(2, 4, kernel_sizes=[5, 3, 1], padding, depth=len(kernel_sizes))

# Get the effective kernel size
print(layer.kernel_size)
```
When `kernel_sizes` is an integer, all proceeding layers are assumed to have kernel size of `1x1` layers. 

#### (2) Forward computation

```python
# Forward pass (expanded form)
layer.train()
y = layer(x)
```

When calling `eval()` the model will automatically reduce the computation graph to its effective single-layer counterpart. 
Forward pass in `eval` mode will use the effective weights instead.

```python
# Forward pass (collapsed form) [automatic]
layer.eval()
y = layer(x)
```

You can access the effective weights as follows:

```python
print(layer.weight)
print(layer.bias)
```

#### (3) Automatic conversion

```python
import torchvision.models as models
from overparam.utils import overparameterize

model = models.alexnet() # Replace this with YOUR_PYTORCH_MODEL()
model = overparameterize(model, depth=2)
```

#### (4) Batch-norm and Residual connections
We also provide support for batch-norm and linear residual connections

- batch-normalization (pseudo-linera layer: linear during `eval` mode)
```python
layer = OverparamConv2d(32, 32, kernel_sizes=3, padding=1, depth=2, 
                        batch_norm=True)
```

- residual-connection 
```python
# every 2 layers, a residual connection is added
layer = OverparamConv2d(32, 32, kernel_sizes=3, padding=1, depth=2,
                        residual=True, residual_intervals=2)
```

- multiple residual connection
```python
# every modulo [1, 2, 3] layers, a residual connection is added
layer = OverparamConv2d(32, 32, kernel_sizes=3, padding=1, depth=2, 
                        residual=True, residual_intervals=[1, 2, 3])
```

- batch-norm and residual connection 
```python
# mimics `BasicBlock` in ResNets
layer = OverparamConv2d(32, 32, kernel_sizes=3, padding=1, depth=2, 
                        batch_norm=True, residual=True, residual_intervals=2)
```


### 3. Cite
To cite the paper or the code:

@article{huh2021lowranksimplicity,
  title={The Low-Rank Simplicity Bias in Deep Networks},
  author={Huh, Minyoung and Mobahi, Hossein and Zhang, Richard and Agrawal, Pulkit and Isola, Phillip},
  journal={arXiv},
  year={2021}
}
