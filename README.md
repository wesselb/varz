# [Varz](http://github.com/wesselb/varz)

[![Build](https://travis-ci.org/wesselb/varz.svg?branch=master)](https://travis-ci.org/wesselb/varz)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/varz/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/varz?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/varz)

Painless optimisation of constrained variables in AutoGrad, TensorFlow, and
PyTorch

_Note:_ Varz requires Python 3.6 or higher and TensorFlow 2 if TensorFlow is 
used.

* [Installation](#installation)
* [Manual](#manual)
    - [Basics](#basics)
    - [Naming](#naming)
    - [Constrained Variables](#constrained-variables)
    - [Automatic Naming of Variables](#automatic-naming-of-variables)
    - [Optimisers](#optimisers)
    - [PyTorch Specifics](#pytorch-specifics)
    - [Getting and Setting Variables as a Vector](#getting-and-setting-variables-as-a-vector)
    - [Getting Variables from a Source](#get-variables-from-a-source)
 * [Examples](#examples)
    - [AutoGrad](#autograd)
    - [TensorFlow](#tensorflow)
    - [PyTorch](#pytorch)
    
## Installation

Before installing the package, please ensure that `gcc` and `gfortran` are 
available.
On OS X, these are both installed with `brew install gcc`;
users of Anaconda may want to instead consider `conda install gcc`.
On Linux, `gcc` is most likely already available, and `gfortran` can be
installed with `apt-get install gfortran`.
Then simply

```
pip install varz
```

## Manual

### Basics
```python
from varz import Vars
```

To begin with, create a *variable container* of the right data type.
For use with AutoGrad, use a `np.*` data type;
for use with PyTorch, use a `torch.*` data type;
and for use with TensorFlow, use a `tf.*` data type.
In this example we'll use AutoGrad.

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
This asserts that a variable with a name `x` already exists, and will throw a
`KeyError` otherwise.

```python
>>> vs['x']
array([[ 1.04404354, -1.98478763],
       [ 1.14176728, -3.2915562 ]])
       
>>> vs['y']
KeyError: 'y'
```

The value of `x` can be changed by assigning it a different value.

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
This means that non-differentiable assignment will also mutate the original;
differentiable assignment, however, will not.

### Naming

Variables may be organised by naming them hierarchically using `/`s. 
For example, `group1/bar`, `group1/foo`, and `group2/bar`.
This is helpful for extracting collections of variables, where wildcards may 
be used to match names.
For example, `*/bar` would match `group1/bar` and `group2/bar`, and 
`group1/*` would match `group1/bar` and `group1/foo`.

The names of all variables can be obtained with `Vars.names`, and variables can 
be printed with `Vars.print`.

Example:


```python
>>> vs = Vars(np.float64)

>>> vs.get(1, name='x1')
array(1.)

>>> vs.get(2, name='x2')
array(2.)

>>> vs.get(3, name='y')
array(3.)

>>> vs.names
['x1', 'x2', 'y']

>>> vs.print()
x1:         1.0
x2:         2.0
y:          3.0
```

### Constrained Variables

* **Positive variables:**
    A variable that is constrained to be *positive* can be created using
    `Vars.positive` or `Vars.pos`.

    ```python
    >>> vs.pos(name='positive_variable')
    0.016925610008314832
    ```

* **Bounded variables:**
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

### Automatic Naming of Variables

To parametrise functions, a common pattern is the following:

```python
def objective(vs):
    x = vs.get(5, name='x')
    y = vs.get(10, name='y')
    
    return (x * y - 5) ** 2 + x ** 2
```

The names for `x` and `y` are necessary, because otherwise new variables will
 be created and initialised every time `objective` is ran.
Varz offers two ways to not having to specify a name for every variable: 
sequential and parametrised specification.

#### Sequential Specification

Sequential specification can be used if, upon execution of `objective`, 
variables are always obtained in the *same order*.
This means that variables can be identified to their position in this order and
hence be named accordingly.
To use sequential specification, decorate the function with `sequential`.

Example:

```python
from varz import sequential


@sequential
def objective(vs):
    x = vs.get(5)  # Initialise to 5.
    y = vs.get()   # Initialise randomly.
    
    return (x * y - 5) ** 2 + x ** 2
```

```python
>>> vs = Vars(np.float64)

>>> objective(vs)
68.65047879833773

>>> objective(vs)  # Running the objective again reuses the same variables.
68.65047879833773

>>> vs.names
['0', '1']

>>> vs.print()
0:          5.0      # This is `x`.
1:          -0.3214  # This is `y`.
```

#### Parametrised Specification

Sequential specification still suffers from boilerplate code like
`x = vs.get(5)` and `y = vs.get()`.
This is the problem that parametrised specification addresses, which allows 
you to specify variables as *arguments to your function*.
To indicate that an argument of the function is a variable, its type hint must
 be set accordingly, as follows:

* **Unbounded variables:** `def f(vs, x: Unbounded)`.

* **Positive variables:** `def f(vs, x: Positive)`.

* **Bounded variables:** `def f(vs, x: Bounded)` to use the default bound, and
    `def f(vs, x: Bounded(lower=1, upper=10))` otherwise.
    
As can be seen to the above, the variable container must also be an argument 
of the function, because that is where the variables will be extracted from.
An argument can be given an initial variable in the way you would expect:
`def f(vs, x: Unbounded = 5)`.

Variable argument and regular argument can be mixed.
If `f` is called, the variable argument must not be specified, because they will
be extracted.
Regular arguments, however, must be specified.

To use parametrised specification, decorate the function with `parametrised`.

Example:

```python
from varz import parametrised, Unbounded, Bounded


@parametrised
def objective(vs, x: Unbounded, y: Bounded(lower=1, upper=3) = 2, option=None):
    print('Option:', option)
    return (x * y - 5) ** 2 + x ** 2
```

```python
>>> vs = Vars(np.float64)

>>> objective(vs)
Option: None
9.757481795615316

>>> objective(vs, 'other')
Option: other
9.757481795615316

>>> objective(vs, option='other')
Option: other
9.757481795615316

>>> objective(vs, x=5)  # This is not valid, because `x` will be obtained automatically.
ValueError: 1 keyword argument(s) not parsed: x.

>>> vs.print()
x:          1.025
y:          2.0
```

### Optimisers

The following optimisers are available:

```
varz.{autograd,tensorflow,torch}.minimise_l_bfgs_b (L-BFGS-B)
varz.{autograd,tensorflow,torch}.minimise_adam     (Adam)
```

The L-BFGS-B algorithm is recommended for deterministic objectives, and Adam
is recommended for stochastic objectives.

See the examples for an illustration how these optimisers can be used.

### PyTorch Specifics

All the variables held by a container can be detached from the current 
computation graph with `Vars.detach` .
To make a copy of the container with detached versions of the variables, use
`Vars.copy` with `detach=True` instead.
Whether variables require gradients can be configured with `Vars.requires_grad`.
By default, no variable requires a gradient.

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

## Examples
The follow examples show how a function can be minimised using the L-BFGS-B
algorithm.

### AutoGrad

```python
import autograd.numpy as np
from varz.autograd import Vars, minimise_l_bfgs_b

target = 5. 


def objective(vs):
    # Get a variable named "x", which must be positive, initialised to 10.
    x = vs.pos(10., name='x')  
    
    return (x ** .5 - target) ** 2  
```

```python
>>> vs = Vars(np.float64)

>>> minimise_l_bfgs_b(objective, vs)
3.17785950743424e-19  # Final objective function value.

>>> vs['x'] - target ** 2
-5.637250666268301e-09
```

### TensorFlow

```python
import tensorflow as tf
from varz.tensorflow import Vars, minimise_l_bfgs_b

target = 5.


def objective(vs):
    # Get a variable named "x", which must be positive, initialised to 10.
    x = vs.pos(10., name='x')  
    
    return (x ** .5 - target) ** 2  
```

```python
>>> vs = Vars(tf.float64)

>>> minimise_l_bfgs_b(objective, vs)
3.17785950743424e-19  # Final objective function value.

>>> vs['x'] - target ** 2
<tf.Tensor: id=562, shape=(), dtype=float64, numpy=-5.637250666268301e-09>
```

### PyTorch

```python
import torch
from varz.torch import Vars, minimise_l_bfgs_b

target = torch.tensor(5., dtype=torch.float64)


def objective(vs):
    # Get a variable named "x", which must be positive, initialised to 10.
    x = vs.pos(10., name='x')  
    
    return (x ** .5 - target) ** 2  
```

```python
>>> vs = Vars(torch.float64)

>>> minimise_l_bfgs_b(lambda v: objective(v['x']), vs, names=['x'])
array(3.17785951e-19)  # Final objective function value.

>>> vs['x'] - target ** 2
tensor(-5.6373e-09, dtype=torch.float64)
```

