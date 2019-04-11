# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
from lab import B
import torch

from . import Vars
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose, approx


def test_get_vars():
    B.backend_to_np()
    vs = Vars(np.int)

    # This test also tests that `Vars.get_vars` always returns the collection
    # of variables in the right order. This is important for optimisation.

    # Initialise some variables.
    vs.get(1, name='a')
    vs.get(2, name='b', group=1)
    vs.get(3, name='c', group=2)
    vs.get(4, name='d', group=2)

    # Test getting all.
    yield eq, vs.get_vars(), [1, 2, 3, 4]

    # Test names.
    yield eq, vs.get_vars('a'), [1]
    yield eq, vs.get_vars('a', 'b'), [1, 2]
    yield eq, vs.get_vars('c', 'a'), [1, 3]
    yield eq, vs.get_vars('c', 'b', 'a'), [1, 2, 3]

    # Test groups.
    yield eq, vs.get_vars(groups=[1]), [2]
    yield eq, vs.get_vars(groups=[2]), [3, 4]
    yield eq, vs.get_vars(groups=[1, 2]), [2, 3, 4]

    # Test names and groups.
    yield eq, vs.get_vars('b', groups=[1]), [2]
    yield eq, vs.get_vars('a', groups=[2]), [1, 3, 4]
    yield eq, vs.get_vars('a', 'd', groups=[2]), [1, 3, 4]
    yield eq, vs.get_vars('d', 'c', 'a', groups=[1]), [1, 2, 3, 4]


def test_get_set_vector():
    B.backend_to_np()
    vs = Vars()

    # Test stacking a matrix and a vector.
    vs.get(shape=(2,), name='a', init=[1, 2])
    vs.get(shape=(2, 2), name='b', init=np.array([[3, 4], [5, 6]]))
    yield allclose, vs.get_vector('a', 'b'), [1, 2, 3, 4, 5, 6]

    # Test setting elements.
    vs.set_vector([6, 5, 4, 3, 2, 1], 'a', 'b')
    yield allclose, vs['a'], [6, 5]
    yield allclose, vs['b'], np.array([[4, 3], [2, 1]])


def test_get_and_init_tf():
    B.backend_to_tf()
    s = tf.Session()

    # Test `float32`.
    vs = Vars(np.float32)
    a = vs.get(1., name='a')
    b = vs.get()
    yield eq, len(vs.vars), 2
    yield eq, a.dtype.as_numpy_dtype, np.float32
    yield eq, b.dtype.as_numpy_dtype, np.float32
    vs.init(s)
    yield eq, s.run(vs['a']), 1.

    # Test `float64`.
    vs = Vars(np.float64)
    a = vs.get(1., name='a')
    b = vs.get()
    yield eq, len(vs.vars), 2
    yield eq, a.dtype.as_numpy_dtype, np.float64
    yield eq, b.dtype.as_numpy_dtype, np.float64
    vs.init(s)
    yield eq, s.run(vs['a']), 1.


def test_positive():
    B.backend_to_np()
    vs = Vars()
    for _ in range(10):
        yield ge, vs.pos(), 0


def test_bounded():
    B.backend_to_np()
    vs = Vars()
    for _ in range(10):
        v = vs.bnd(lower=10, upper=11)
        yield ge, v, 10
        yield le, v, 11


def test_assignment():
    B.backend_to_np()
    vs = Vars()

    # Generate some variables.
    vs.get(1., name='unbounded')
    vs.pos(2., name='positive')
    vs.bnd(3., lower=0, upper=10, name='bounded')

    # Check that they have the right values.
    yield eq, 1., vs['unbounded']
    yield allclose, 2., vs['positive']
    yield allclose, 3., vs['bounded']

    # Assign some new values.
    vs.assign('unbounded', 4.)
    vs.assign('positive', 5.)
    vs.assign('bounded', 6.)

    # Again check that they have the right values.
    yield eq, 4., vs['unbounded']
    yield allclose, 5., vs['positive']
    yield allclose, 6., vs['bounded']


def test_detach_torch():
    B.backend_to_torch()
    vs = Vars(torch.float64)

    # Create a variable and copy variable storage.
    vs.pos(1, name='a')
    vs2 = vs.detach()

    # Require gradients for both.
    vs.requires_grad(True)
    vs2.requires_grad(True)

    # Do a backward pass.
    (vs2['a'] ** 2).backward()

    # Check that values are equal, but gradients only computed for one.
    yield eq, vs['a'], 1
    yield eq, vs.get_vars('a')[0].grad, None
    yield eq, vs2['a'], 1
    yield eq, vs2.get_vars('a')[0].grad, 2
