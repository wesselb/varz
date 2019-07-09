# [Varz](http://github.com/wesselb/varz)

[![Build](https://travis-ci.org/wesselb/varz.svg?branch=master)](https://travis-ci.org/wesselb/varz)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/varz/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/varz?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/varz)

Painless optimisation of constrained variables in PyTorch, TensorFlow, and 
AutoGrad

_Note:_ Varz requires TensorFlow 2.

* [Installation](#installation)
* [Manual](#manual)
    - [Naming](#naming)
    - [Constrained Variables](#constrained-variables)
    - [Getting and Setting Variables as a Vector](#getting-and-setting-variables-as-a-vector)
    - [AutoGrad](#autograd)
    - [TensorFlow](#tensorflow)
    - [PyTorch](#pytorch)
    
## Installation

Before installing the package, please ensure that `gcc` and `gfortran` are 
available.
On OS X, these are both installed with `brew install gcc`;
users of Anaconda may want to instead consider `conda install gcc`.
Then simply

```
pip install varz
```

## Manual

```python
from varz import Vars
```

To begin with, create a variable container of the right data type.
For use with NumPy and AutoGrad, use a `np.*` data type;
for use with PyTorch, use a `torch.*` data type;
and for use with TensorFlow, use a `tf.*` data type.
In this example we'll use NumPy and AutoGrad.

```python
>>> vs = Vars(np.float64)
```

Now a variable can be created by requesting it, giving it an initial value and
a name.
 
```python
>>> vs.get(np.random.randn(2, 2), name='x')
array([[ 1.04404354, -1.98478763],
       [ 1.14176728, -3.2915562 ]])
```

If the same variable is created again, because a variable with the name `x` 
already exists, the existing variable will be returned.

```python
>>> vs.get(name='x')
array([[ 1.04404354, -1.98478763],
       [ 1.14176728, -3.2915562 ]])
```

Alternatively, indexing syntax may be used to get the existing variable `x`.

```python
>>> vs['x']
array([[ 1.04404354, -1.98478763],
       [ 1.14176728, -3.2915562 ]])
```

The value of `x` may be changed by assigning it a different value.

```python
>>> vs.assign('x', np.random.randn(2, 2))
array([[ 1.43477728,  0.51006941],
       [-0.74686452, -1.05285767]])
```

By default, assignment is non-differentiable and _overwrites_ data.
For differentiable assignment, which _replaces_ data, set the keyword argument 
`differentiable=True`.

```python
>>> vs.assign('x', np.random.randn(2, 2), differentiable=True)
array([[ 0.12500578, -0.21510423],
       [-0.61336039,  1.23074066]])
```

The variable container can be copied with `vs.copy()`.
Note that the copy _shares its variables with the original_.
This means that assignment will also mutate the original;
differentiable assignment, however, will not.

### Naming

Variables may be organised by naming them hierarchically using `/`s. 
For example, `group1/bar`, `group1/foo`, and `group2/bar`.
This is helpful for extracting collections of variables, where wildcards may 
be used to match names.
For example, `*/bar` would match `group1/bar` and `group2/bar`, and 
`group1/*` would match `group1/bar` and `group1/foo`.

### Constrained Variables

A variable that is constrained to be *positive* can be created using
`Vars.positive` or `Vars.pos`.

```python
>>> vs.pos(name='positive_variable')
0.016925610008314832
```

A variable that is constrained to be *bounded* can be created using
`Vars.bounded` or `Vars.bnd`.

```python
>>> vs.bnd(name='bounded_variable', lower=1, upper=2)
1.646772663807718
```

These constrained variables are created by transforming some *latent 
unconstrained representation* to the desired constrained space.
The latent variables can be obtained using `Vars.get_vars`.

```python
>>> vs.get_vars('positive_variable', 'bounded_variable')
[array(-4.07892742), array(-0.604883)]
```

To illustrate the use of wildcards, the following is equivalent:

```python
>>> vs.get_vars('*_variable')
[array(-4.07892742), array(-0.604883)]
```

### Getting and Setting Variables as a Vector

It may be desirable to get the latent representations of a collection of 
variables as a single vector, e.g. when feeding them to an optimiser.
This can be achieved with `Vars.get_vector`.

```python
>>> vs.get_vector('x', '*_variable')
array([ 0.12500578, -0.21510423, -0.61336039,  1.23074066, -4.07892742,
       -0.604883  ])
```

Similarly, to update the latent representation of a collection of variables,
`Vars.set_vector` can be used.

```python
>>> vs.set_vector(np.ones(6), 'x', '*_variable')
[array([[1., 1.],
        [1., 1.]]), array(1.), array(1.)]

>>> vs.get_vector('x', '*_variable')
array([1., 1., 1., 1., 1., 1.])
```

### AutoGrad

The function `varz.autograd.minimise_l_bfgs_b` can be used to perform 
minimisation using the L-BFGS-B algorithm.

Example of optimising variables:

```python
import autograd.numpy as np
from varz.autograd import Vars, minimise_l_bfgs_b

target = 5. 


def objective(x):  # `x` must be positive!
    return (x ** .5 - target) ** 2  
```

```python
>>> vs = Vars(np.float64)

>>> vs.pos(10., name='x')
10.000000000000002

>>> minimise_l_bfgs_b(lambda v: objective(v['x']), vs, names=['x'])
3.17785950743424e-19  # Final objective function value.

>>> vs['x'] - target ** 2
-5.637250666268301e-09
```

### TensorFlow

The function `varz.tensorflow.minimise_l_bfgs_b` can be used to perform 
minimisation using the L-BFGS-B algorithm.

Example of optimising variables:

```python
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b

target = 5.


def objective(x):  # `x` must be positive!
    return (x ** .5 - target) ** 2  
```

```python
>>> vs = Vars(tf.float64)

>>> vs.pos(10., name='x')
<tf.Tensor: id=11, shape=(), dtype=float64, numpy=10.000000000000002>

>>> minimise_l_bfgs_b(lambda v: objective(v['x']), vs, names=['x'])
3.17785950743424e-19  # Final objective function value.

>>> vs['x'] - target ** 2
<tf.Tensor: id=562, shape=(), dtype=float64, numpy=-5.637250666268301e-09>
```

### PyTorch

All the variables held by a container can be detached from the current 
computation graph with `Vars.detach` .
To make a copy of the container with detached versions of the variables, use
`Vars.copy` with `detach=True` instead.
Whether variables require gradients can be configured with `Vars.requires_grad`.
By default, no variable requires a gradient.

The function `varz.torch.minimise_l_bfgs_b` can be used to perform minimisation 
using the L-BFGS-B algorithm.

Example of optimising variables:

```python
import torch
from varz.torch import Vars, minimise_l_bfgs_b


target = torch.tensor(5., dtype=torch.float64)


def objective(x):  # `x` must be positive!
    return (x ** .5 - target) ** 2
```

```python
>>> vs = Vars(torch.float64)

>>> vs.pos(10., name='x')
tensor(10.0000, dtype=torch.float64)

>>> minimise_l_bfgs_b(lambda v: objective(v['x']), vs, names=['x'])
array(3.17785951e-19)  # Final objective function value.

>>> vs['x'] - target ** 2
tensor(-5.6373e-09, dtype=torch.float64)
```


### Get Variables from a Source

The keyword argument `source` can set to a tensor from which the latent 
variables will be obtained.

Example:

```python
>>> vs = Vars(np.float32, source=np.array([1, 2, 3, 4, 5]))

>>> vs.get()
array(1., dtype=float32)

>>> vs.get(shape=(3,))
array([2., 3., 4.], dtype=float32)

>>> vs.pos()
148.41316

>>> np.exp(5).astype(np.float32)
148.41316
```
