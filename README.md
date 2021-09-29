# [Varz](http://github.com/wesselb/varz)

[![CI](https://github.com/wesselb/varz/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/varz/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/varz/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/varz?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/varz)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Painless optimisation of constrained variables in AutoGrad, TensorFlow, PyTorch, and JAX

* [Requirements and Installation](#requirements-and-installation)
* [Manual](#manual)
    - [Basics](#basics)
    - [Naming](#naming)
    - [Constrained Variables](#constrained-variables)
    - [Automatic Naming of Variables](#automatic-naming-of-variables)
        - [Sequential Specification](#sequential-specification)
        - [Parametrised Specification](#parametrised-specification)
        - [Namespaces](#namespaces)
        - [Structlike Specification](#structlike-specification)
    - [Optimisers](#optimisers)
    - [PyTorch Specifics](#pytorch-specifics)
    - [Getting and Setting Latent Representations of Variables as a Vector](#getting-and-setting-latent-representations-of-variables-as-a-vector)
    - [Getting Variables from a Source](#get-variables-from-a-source)
 * [Examples](#examples)
    - [AutoGrad](#autograd)
    - [TensorFlow](#tensorflow)
    - [PyTorch](#pytorch)
    - [JAX](#jax)
    
## Requirements and Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```bash
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
for use with TensorFlow, use a `tf.*` data type;
and for use with JAX, use a `jnp.*` data type.
In this example we'll use AutoGrad.

```python
>>> vs = Vars(np.float64)
```

Now a variable can be created by requesting it, giving it an initial value and
a name.
 
```python
>>> vs.unbounded(np.random.randn(2, 2), name="x")
array([[ 1.04404354, -1.98478763],
       [ 1.14176728, -3.2915562 ]])
```

If the same variable is created again, because a variable with the name `x` 
already exists, the existing variable will be returned, even if you again pass it an
initial value.

```python
>>> vs.unbounded(np.random.randn(2, 2), name="x")
array([[ 1.04404354, -1.98478763],
       [ 1.14176728, -3.2915562 ]])

>>> vs.unbounded(name="x")
array([[ 1.04404354, -1.98478763],
       [ 1.14176728, -3.2915562 ]])
```

Alternatively, indexing syntax may be used to get the existing variable `x`.
This asserts that a variable with the name `x` already exists and will throw a
`KeyError` otherwise.

```python
>>> vs["x"]
array([[ 1.04404354, -1.98478763],
       [ 1.14176728, -3.2915562 ]])
       
>>> vs["y"]
KeyError: 'y'
```

The value of `x` can be changed by assigning it a different value.

```python
>>> vs.assign("x", np.random.randn(2, 2))
array([[ 1.43477728,  0.51006941],
       [-0.74686452, -1.05285767]])
```

By default, assignment is non-differentiable and _overwrites_ data.
The variable can be deleted by passing its name to `vs.delete`:

```python
>>> vs.delete("x")

>>> vs["x"]
KeyError: 'x'
```

When a variable is first created, you can set the keyword argument `visible`
to `False` if you want to make the variable invisible to the
variable-aggregating operations [`vs.get_latent_vars`](#constrained-variables)
and
[`vs.get_latent_vector`](#getting-and-setting-latent-representations-of-variables-as-a-vector).
These variable-aggregating operations are used in optimisers to get the intended
collection of variable to optimise.
Therefore, setting `visible` to `False` will prevent a variable from being
optimised.

Finally, a variable container can be copied with `vs.copy()`.
Copies are lightweight and _share their variables with the originals_.
As a consequence, however, assignment in a copy will also mutate the original.
[Differentiable assignment, however, will not.](#differentiable-assignment)


### Naming

Variables may be organised by naming them hierarchically using `.`s. 
For example, you could name like `group1.bar`, `group1.foo`, and `group2.bar`.
This is helpful for extracting collections of variables, where wildcards may 
be used to match names.
For example, `*.bar` would match `group1.bar` and `group2.bar`, and 
`group1.*` would match `group1.bar` and `group1.foo`.
See also [here](#getting-and-setting-latent-representations-of-variables-as-a-vector).

The names of all variables can be obtained with `Vars.names`, and variables can 
be printed with `Vars.print`.

Example:

```python
>>> vs = Vars(np.float64)

>>> vs.unbounded(1, name="x1")
array(1.)

>>> vs.unbounded(2, name="x2")
array(2.)

>>> vs.unbounded(3, name="y")
array(3.)

>>> vs.names
['x1', 'x2', 'y']

>>> vs.print()
x1:         1.0
x2:         2.0
y:          3.0
```

### Constrained Variables

* **Unbounded variables:**
  A variable that is unbounded can be created using
  `Vars.unbounded` or `Vars.ubnd`.

    ```python
    >>> vs.ubnd(name="normal_variable")
    0.016925610008314832
    ```

* **Positive variables:**
    A variable that is constrained to be *positive* can be created using
    `Vars.positive` or `Vars.pos`.

    ```python
    >>> vs.pos(name="positive_variable")
    0.016925610008314832
    ```

* **Bounded variables:**
    A variable that is constrained to be *bounded* can be created using
    `Vars.bounded` or `Vars.bnd`.

    ```python
    >>> vs.bnd(name="bounded_variable", lower=1, upper=2)
    1.646772663807718
    ```
  
* **Lower-triangular matrix:**
    A matrix variable that is constrained to be *lower triangular* can be
    created using `Vars.lower_triangular` or `Vars.tril`. Either an
    initialisation or a shape of square matrix must be given.
    
    ```python
    >>> vs.tril(shape=(2, 2), name="lower_triangular")
    array([[ 2.64204459,  0.        ],
           [-0.14055559, -1.91298679]])
    ```
  
* **Positive-definite matrix:**
    A matrix variable that is contrained to be *positive definite* can be
    created using `Vars.positive_definite` or `Vars.pd`. Either an
    initialisation or a shape of square matrix must be given.
    
    ```python
    >>> vs.pd(shape=(2, 2), name="positive_definite")
    array([[ 1.64097496, -0.52302151],
           [-0.52302151,  0.32628302]])
    ```
  
* **Orthogonal matrix:**
    A matrix variable that is constrained to be *orthogonal* can be created using
    `Vars.orthogonal` or `Vars.orth`. Either an initialisation or a
    shape of square matrix must be given.
    
    ```python
    >>> vs.orth(shape=(2, 2), name="orthogonal")
    array([[ 0.31290403, -0.94978475],
           [ 0.94978475,  0.31290403]])
    ```

These constrained variables are created by transforming some *latent 
unconstrained representation* to the desired constrained space.
The latent variables can be obtained using `Vars.get_latent_vars`.

```python
>>> vs.get_latent_vars("positive_variable", "bounded_variable")
[array(-4.07892742), array(-0.604883)]
```

To illustrate the use of wildcards, the following is equivalent:

```python
>>> vs.get_latent_vars("*_variable")
[array(-4.07892742), array(-0.604883)]
```

Variables can be excluded by prepending a dash:

```python
>>> vs.get_latent_vars("*_variable", "-bounded_*")
[array(-4.07892742)]
```

### Automatic Naming of Variables

To parametrise functions, a common pattern is the following:

```python
def objective(vs):
    x = vs.unbounded(5, name="x")
    y = vs.unbounded(10, name="y")
    
    return (x * y - 5) ** 2 + x ** 2
```

The names for `x` and `y` are necessary, because otherwise new variables will
 be created and initialised every time `objective` is run.
Varz offers two ways to not having to specify a name for every variable: 
sequential and parametrised specification.

#### Sequential Specification

Sequential specification can be used if, upon execution of `objective`, 
variables are always obtained in the *same order*.
This means that variables can be identified with their position in this order
and hence be named accordingly.
To use sequential specification, decorate the function with `sequential`.

Example:

```python
from varz import sequential

@sequential
def objective(vs):
    x = vs.unbounded(5)  # Initialise to 5.
    y = vs.unbounded()   # Initialise randomly.
    
    return (x * y - 5) ** 2 + x ** 2
```

```python
>>> vs = Vars(np.float64)

>>> objective(vs)
68.65047879833773

>>> objective(vs)  # Running the objective again reuses the same variables.
68.65047879833773

>>> vs.names
['var0', 'var1']

>>> vs.print()
var0:       5.0      # This is `x`.
var1:       -0.3214  # This is `y`.
```

#### Parametrised Specification

Sequential specification still suffers from boilerplate code like
`x = vs.unbounded(5)` and `y = vs.unbounded()`.
This is the problem that parametrised specification addresses, which allows 
you to specify variables as *arguments to your function*.
Import `from varz.spec import parametrised`.
To indicate that an argument of the function is a variable, as opposed to a 
regular argument, the argument's type hint must be set accordingly, as follows:

* **Unbounded variables:**
    ```python
    @parametrised
    def f(vs, x: Unbounded):
        ...
    ```

* **Positive variables:**
    ```python
    @parametrised
    def f(vs, x: Positive):
        ...
    ```

* **Bounded variables:**
    The following two specifications are possible. The former uses the
    default bounds and the latter uses specified bounds.
     
    ```python
    @parametrised
    def f(vs, x: Bounded):
        ...
    ```
  
    ```python
    @parametrised
    def f(vs, x: Bounded(lower=1, upper=10)):
        ...
    ```
    
* **Lower-triangular variables:**
    ```python
    @parametrised
    def f(vs, x: LowerTriangular(shape=(2, 2))):
        ...
    ```

* **Positive-definite variables:**
    ```python
    @parametrised
    def f(vs, x: PositiveDefinite(shape=(2, 2))):
        ...
    ```
  
* **Orthogonal variables:**
    ```python
    @parametrised
    def f(vs, x: Orthogonal(shape=(2, 2))):
        ...
    ```
    
As can be seen from the above, the variable container must also be an argument 
of the function, because that is where the variables will be obtained from.
A variable can be given an initial value in the way you would expect:
```python
@parametrised
def f(vs, x: Unbounded = 5):
    ...
```

Variable arguments and regular arguments can be mixed.
If `f` is called, variable arguments must not be specified, because they 
will be obtained automatically.
Regular arguments, however, must be specified.

To use parametrised specification, decorate the function with `parametrised`.

Example:

```python
from varz import parametrised, Unbounded, Bounded

@parametrised
def objective(vs, x: Unbounded, y: Bounded(lower=1, upper=3) = 2, option=None):
    print("Option:", option)
    return (x * y - 5) ** 2 + x ** 2
```

```python
>>> vs = Vars(np.float64)

>>> objective(vs)
Option: None
9.757481795615316

>>> objective(vs, "other")
Option: other
9.757481795615316

>>> objective(vs, option="other")
Option: other
9.757481795615316

>>> objective(vs, x=5)  # This is not valid, because `x` will be obtained automatically from `vs`.
ValueError: 1 keyword argument(s) not parsed: x.

>>> vs.print()
x:          1.025
y:          2.0
```

#### Namespaces

Namespaces can be used to group all variables in a function together.

Example:

```python
from varz import namespace

@namespace("test")
def objective(vs):
    x = vs.unbounded(5, name="x")
    y = vs.unbounded(name="y")
    
    return x + y
```

```python
>>> vs = Vars(np.float64)

>>> objective(vs)
6.12448906632577

>>> vs.names
['test.x', 'test.y']

>>> vs.print()
test.x:     5.0
test.y:     1.124
```

You can combine namespace with other specification methods:

```python
from varz import namespace

@namespace("test")
@sequential
def objective(vs):
    x = vs.unbounded(5)
    y = vs.unbounded()
    
    return x + y
```

```python
>>> vs = Vars(np.float64)

>>> objective(vs)
4.812730329303665

>>> vs.names
['test.var0', 'test.var1']

>>> vs.print()
test.var0:  5.0
test.var1:  -0.1873
```

#### Structlike Specification

For any variable container `vs`, `vs.struct` gives an object which you can treat like
nested struct, list, or dictionary to automatically generate variable names.
For example, `vs.struct.model["a"].variance.positive()` would be equivalent to
`vs.positive(name="model[a].variance")`.
After variables have been defined in this way, they also be extracted via `vs.struct`:
`vs.struct.model["a"].variance()` would be equivalent to `vs["model[a].variance"]`.

Example:

```python
def objective(vs):
    params = vs.struct
    
    x = params.x.unbounded()
    y = params.y.unbounded()
    
    for model_params, model in zip(params.models, [object(), object(), object()]):
        model_params.specific_parameter1.positive()
        model_params.specific_parameter2.positive()
    
    return x + y
```

```python
>>> vs = Vars(np.float64)

>>> objective(vs)
-0.08322955725015702

>>> vs.names
['x',
 'y',
 'models[0].specific_parameter1',
 'models[0].specific_parameter2',
 'models[1].specific_parameter1',
 'models[1].specific_parameter2',
 'models[2].specific_parameter1',
 'models[2].specific_parameter2']

>>> vs.print()
x:          -0.8963
y:          0.8131
models[0].specific_parameter1: 0.01855
models[0].specific_parameter2: 0.6644
models[1].specific_parameter1: 0.3542
models[1].specific_parameter2: 0.3642
models[2].specific_parameter1: 0.5807
models[2].specific_parameter2: 0.5977

>>> vs.struct.models[0].specific_parameter1()
0.018551827512328086

>>> vs.struct.models[0].specific_parameter2()
0.6643533007198247
```

There are a few methods available for convenient manipulation of the variable struct.
In the following, let `params = vs.struct`.

* _Go up a directory_:
    `params.a.b.c.up()` goes up one directory and gives `params.a.b`.
    If you want to be sure about which directory you are going up, you can pass
    the name of the directory you want to go up as an argument:
    `params.a.b.c.up("c")` will give the intended result, but
    `params.a.b.c.up("b")` will result in an assertion error.
* _Get all variables in a path_:
    `params.a.all()` gives the regex `a.*`.
* _Check if a variable exists_:
    `bool(params.a)` gives `True` if `a` is a defined variable and `False`
    otherwise.
* _Assign a value to a variable_:
    `params.a.assign(1)` assigns `1` to `a`.
* _Delete a variable_:
    `params.a.delete()` deletes `a`.


### Optimisers

The following optimisers are available:

```
varz.{autograd,tensorflow,torch,jax}.minimise_l_bfgs_b (L-BFGS-B)
varz.{autograd,tensorflow,torch,jax}.minimise_adam     (ADAM)
```

The L-BFGS-B algorithm is recommended for deterministic objectives and ADAM
is recommended for stochastic objectives.

See the examples for an illustration of how these optimisers can be used.

### PyTorch Specifics

All the variables held by a container can be detached from the current 
computation graph with `Vars.detach` .
To make a copy of the container with detached versions of the variables, use
`Vars.copy` with `detach=True` instead.
Whether variables require gradients can be configured with `Vars.requires_grad`.
By default, no variable requires a gradient.

### Getting and Setting Latent Representations of Variables as a Vector

It may be desirable to get the latent representations of a collection of 
variables as a single vector, e.g. when feeding them to an optimiser.
This can be achieved with `Vars.get_latent_vector`.

```python
>>> vs.get_latent_vector("x", "*_variable")
array([0.12500578, -0.21510423, -0.61336039, 1.23074066, -4.07892742,
       -0.604883])
```

Similarly, to update the latent representation of a collection of variables,
`Vars.set_latent_vector` can be used.

```python
>>> vs.set_latent_vector(np.ones(6), "x", "*_variable")
[array([[1., 1.],
        [1., 1.]]), array(1.), array(1.)]

>>> vs.get_latent_vector("x", "*_variable")
array([1., 1., 1., 1., 1., 1.])
```

#### Differentiable Assignment
By default, `Vars.set_latent_vector` will overwrite the variables, just like
`Vars.assign`.
This has as an unfortunate consequence that you cannot differentiate with respect to
the assigned values.
To be able to differentiable with respect to the assigned values, set the keyword
`differentiable=True` in the call to `Vars.set_latent_vector`.
Unlike regular assignment, if the variable container is a copy of some original,
differentiable assignment will not mutate the variables in the original.

### Get Variables from a Source

The keyword argument `source` can set to a tensor from which the latent 
variables will be obtained.

Example:

```python
>>> vs = Vars(np.float32, source=np.array([1, 2, 3, 4, 5]))

>>> vs.unbounded()
array(1., dtype=float32)

>>> vs.unbounded(shape=(3,))
array([2., 3., 4.], dtype=float32)

>>> vs.pos()
148.41316

>>> np.exp(5).astype(np.float32)
148.41316
```

## GPU Support
To create and optimise variables on a GPU,
[set the active device to a GPU](https://github.com/wesselb/lab#devices).
The easiest way of doing this is to `import lab as B` and
`B.set_global_device("gpu:0")`.

## Examples
The follow examples show how a function can be minimised using the L-BFGS-B
algorithm.

### AutoGrad

```python
import autograd.numpy as np
from varz.autograd import Vars, minimise_l_bfgs_b

target = 5.0 


def objective(vs):
    # Get a variable named "x", which must be positive, initialised to 10.
    x = vs.pos(10.0, name="x")  
    return (x ** 0.5 - target) ** 2  
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

target = 5.0


def objective(vs):
    # Get a variable named "x", which must be positive, initialised to 10.
    x = vs.pos(10.0, name="x")  
    return (x ** 0.5 - target) ** 2  
```

```python
>>> vs = Vars(tf.float64)

>>> minimise_l_bfgs_b(objective, vs)
3.17785950743424e-19  # Final objective function value.

>>> vs['x'] - target ** 2
<tf.Tensor: id=562, shape=(), dtype=float64, numpy=-5.637250666268301e-09>

>>> vs = Vars(tf.float64)

>>> minimise_l_bfgs_b(objective, vs, jit=True)  # Speed up optimisation with TF's JIT!
3.17785950743424e-19
```

### PyTorch

```python
import torch
from varz.torch import Vars, minimise_l_bfgs_b

target = torch.tensor(5.0, dtype=torch.float64)


def objective(vs):
    # Get a variable named "x", which must be positive, initialised to 10.
    x = vs.pos(10.0, name="x")  
    return (x ** 0.5 - target) ** 2  
```

```python
>>> vs = Vars(torch.float64)

>>> minimise_l_bfgs_b(objective, vs)
array(3.17785951e-19)  # Final objective function value.

>>> vs["x"] - target ** 2
tensor(-5.6373e-09, dtype=torch.float64)

>>> vs = Vars(torch.float64)

>>> minimise_l_bfgs_b(objective, vs, jit=True)  # Speed up optimisation with PyTorch's JIT!
array(3.17785951e-19)
```

### JAX

```python
import jax.numpy as jnp
from varz.jax import Vars, minimise_l_bfgs_b

target = 5.0


def objective(vs):
    # Get a variable named "x", which must be positive, initialised to 10.
    x = vs.pos(10.0, name="x")  
    return (x ** 0.5 - target) ** 2  
```

```python
>>> vs = Vars(jnp.float64)

>>> minimise_l_bfgs_b(objective, vs)
array(3.17785951e-19)  # Final objective function value.

>>> vs["x"] - target ** 2
-5.637250666268301e-09

>>> vs = Vars(jnp.float64)

>>> minimise_l_bfgs_b(objective, vs, jit=True)  # Speed up optimisation with Jax's JIT!
array(3.17785951e-19)
```


