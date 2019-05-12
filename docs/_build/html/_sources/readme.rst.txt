`Varz <http://github.com/wesselb/varz>`__
=========================================

|Build| |Coverage Status| |Latest Docs|

Painless optimisation of constrained variables in PyTorch, TensorFlow,
and AutoGrad

-  `Installation <#installation>`__
-  `Manual <#manual>`__

   -  `Naming <#naming>`__
   -  `Constrained Variables <#constrained-variables>`__
   -  `Getting and Setting Variables as a
      Vector <#getting-and-setting-variables-as-a-vector>`__
   -  `AutoGrad <#autograd>`__
   -  `TensorFlow <#tensorflow>`__
   -  `PyTorch <#pytorch>`__

Installation
------------

The package is tested for Python 2.7 and Python 3.6, which are the
versions recommended to use. To install the package, use the following
sequence of commands:

::

    git clone https://github.com/wesselb/varz
    cd varz
    make install

Manual
------

.. code:: python

    from varz import Vars

To begin with, create a variable container of the right data type. For
use with NumPy and AutoGrad, use a ``np.*`` data type; for use with
PyTorch, use a ``torch.*`` data type; and for use with TensorFlow, use a
``tf.*`` data type. In this example we'll use NumPy and AutoGrad.

.. code:: python

    >>> vs = Vars(np.float64)

Now a variable can be created by requesting it, giving it an initial
value and a name.

.. code:: python

    >>> vs.get(np.random.randn(2, 2), name='x')
    array([[ 1.04404354, -1.98478763],
           [ 1.14176728, -3.2915562 ]])

If the same variable is created again, because a variable with the name
``x`` already exists, the existing variable will be returned.

.. code:: python

    >>> vs.get(name='x')
    array([[ 1.04404354, -1.98478763],
           [ 1.14176728, -3.2915562 ]])

Alternatively, indexing syntax may be used to get the existing variable
``x``.

.. code:: python

    >>> vs['x']
    array([[ 1.04404354, -1.98478763],
           [ 1.14176728, -3.2915562 ]])

The value of ``x`` may be changed by assigning it a different value.

.. code:: python

    >>> vs.assign('x', np.random.randn(2, 2))
    array([[ 1.43477728,  0.51006941],
           [-0.74686452, -1.05285767]])

By default, assignment is non-differentiable and overwrites data. For
differentiable assignment, set the keyword argument
``differentiable=True``.

.. code:: python

    >>> vs.assign('x', np.random.randn(2, 2), differentiable=True)
    array([[ 0.12500578, -0.21510423],
           [-0.61336039,  1.23074066]])

*Note:* In TensorFlow, non-differentiable assignment operations return
tensors that must be run to perform the assignments.

Naming
~~~~~~

Variables may be organised by naming them hierarchically using ``/``\ s.
For example, ``group1/bar``, ``group1/foo``, and ``group2/bar``. This is
helpful for extracting collections of variables, where wildcards may be
used to match names. For example, ``*/bar`` would match ``group1/bar``
and ``group2/bar``, and ``group1/*`` would match ``group1/bar`` and
``group1/foo``.

Constrained Variables
~~~~~~~~~~~~~~~~~~~~~

A variable that is constrained to be *positive* can be created using
``Vars.positive`` or ``Vars.pos``.

.. code:: python

    >>> vs.pos(name='positive_variable')
    0.016925610008314832

A variable that is constrained to be *bounded* can be created using
``Vars.bounded`` or ``Vars.bnd``.

.. code:: python

    >>> vs.bnd(name='bounded_variable', lower=1, upper=2)
    1.646772663807718

These constrained variables are created by transforming some *latent
unconstrained representation* to the desired constrained space. The
latent variables can be obtained using ``Vars.get_vars``.

.. code:: python

    >>> vs.get_vars('positive_variable', 'bounded_variable')
    [array(-4.07892742), array(-0.604883)]

To illustrate the use of wildcards, the following is equivalent:

.. code:: python

    >>> vs.get_vars('*_variable')
    [array(-4.07892742), array(-0.604883)]

Getting and Setting Variables as a Vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

It may be desirable to get the latent representations of a collection of
variables as a single vector, e.g. when feeding them to an optimiser.
This can be achieved with ``Vars.get_vector``.

.. code:: python

    >>> vs.get_vector('x', '*_variable')
    array([ 0.12500578, -0.21510423, -0.61336039,  1.23074066, -4.07892742,
           -0.604883  ])

Similarly, to update the latent representation of a collection of
variables, ``Vars.set_vector`` can be used.

.. code:: python

    >>> vs.set_vector(np.ones(6), 'x', '*_variable')
    [array([[1., 1.],
            [1., 1.]]), array(1.), array(1.)]

    >>> vs.get_vector('x', '*_variable')
    array([1., 1., 1., 1., 1., 1.])

AutoGrad
~~~~~~~~

The function ``varz.numpy.minimise_l_bfgs_b`` can be used to perform
minimisation using the L-BFGS-B algorithm.

Example of optimising variables:

.. code:: python

    import autograd.numpy as np
    from varz import Vars
    from varz.numpy import minimise_l_bfgs_b

    target = 5.


    def objective(x):  # `x` must be positive!
        return (x ** .5 - target) ** 2  

.. code:: python

    >>> vs = Vars(np.float64)

    >>> vs.pos(10., name='x')
    10.000000000000002

    >>> minimise_l_bfgs_b(lambda v: objective(v['x']), vs, names=['x'])
    3.17785950743424e-19  # Final objective function value.

    >>> vs['x'] - target ** 2
    -5.637250666268301e-09

TensorFlow
~~~~~~~~~~

All the variables held by a container can be initialised at once with
``Vars.init``.

Example of optimising variables:

.. code:: python

    import tensorflow as tf
    from tensorflow.contrib.opt import ScipyOptimizerInterface as SOI
    from varz import Vars

    target = tf.constant(5., dtype=tf.float64)

    vs = Vars(tf.float64)
    x = vs.pos(10., name='x')
    objective = (x ** .5 - target) ** 2  # `x` must be positive!

.. code:: python

    >>> opt = SOI(objective, var_list=vs.get_vars('x'))

    >>> sess = tf.Session()

    >>> vs.init(sess)

    >>> opt.minimize(sess)

    >>> sess.run(vs['x'] - target ** 2)
    -5.637250666268301e-09

PyTorch
~~~~~~~

All the variables held by a container can be detached from the current
computation graph with ``Vars.detach_vars``. To make a copy of the
container with detached versions of the variables, use ``Vars.detach``
instead. Whether variables require gradients can be configured with
``Vars.requires_grad``. By default, no variable requires a gradient.

The function ``varz.torch.minimise_l_bfgs_b`` can be used to perform
minimisation using the L-BFGS-B algorithm.

Example of optimising variables:

.. code:: python

    import torch
    from varz import Vars
    from varz.torch import minimise_l_bfgs_b

    target = torch.tensor(5., dtype=torch.float64)


    def objective(x):  # `x` must be positive!
        return (x ** .5 - target) ** 2

.. code:: python

    >>> vs = Vars(torch.float64)

    >>> vs.pos(10., name='x')
    tensor(10.0000, dtype=torch.float64)

    >>> minimise_l_bfgs_b(lambda v: objective(v['x']), vs, names=['x'])
    array(1.36449515e-13)  # Final objective function value.

    >>> vs['x'] - target ** 2
    tensor(1.6378e-07, dtype=torch.float64)

.. |Build| image:: https://travis-ci.org/wesselb/varz.svg?branch=master
   :target: https://travis-ci.org/wesselb/varz
.. |Coverage Status| image:: https://coveralls.io/repos/github/wesselb/varz/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/wesselb/varz?branch=master
.. |Latest Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://wesselb.github.io/varz
