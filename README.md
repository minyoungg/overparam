# Overparam layers
PyTorch linear over-parameterization layers with automatic graph reduction.   

Official codebase used in:

**The Low-Rank Simplicity Bias in Deep Networks**  
[Minyoung Huh](http://minyounghuh.com/) &nbsp; [Hossein Mobahi]() &nbsp; [Richard Zhang](https://richzhang.github.io/) &nbsp; [Brian Cheung]() &nbsp; [Pulkit Agrawal]() &nbsp; [Phillip Isola]()     
MIT CSAIL &nbsp; Google Research &nbsp; Adobe Research &nbsp; MIT BCS   
arXiv 2021   
**[[project page]](https://minyoungg.github.io/overparam/) | [[paper]](https://arxiv.org/abs/2103.10427)**     


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
The layers work exactly the same as any `torch.nn` layers.

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
When `kernel_sizes` is an integer, all proceeding layers are assumed to have kernel size of `1x1`. 

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
We also provide support for batch-norm and linear residual connections.

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
```
@article{huh2021lowranksimplicity,
  title={The Low-Rank Simplicity Bias in Deep Networks},
  author={Huh, Minyoung and Mobahi, Hossein and Zhang, Richard and Agrawal, Pulkit and Isola, Phillip},
  journal={arXiv},
  year={2021}
}
```
