# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
import numpy as np
import pytest
import tensorflow as tf
import torch
from plum import Dispatcher

from varz import Vars
from .util import allclose, approx


def test_get_vars():
    vs = Vars(np.int)

    # This test also tests that `Vars.get_vars` always returns the collection
    # of variables in the right order. This is important for optimisation.

    # Initialise some variables.
    vs.get(1, name='a')
    vs.get(2, name='1/b')
    vs.get(3, name='2/c')
    vs.get(4, name='2/d')

    # Test getting all.
    assert vs.get_vars() == [1, 2, 3, 4]
    assert vs.get_vars(indices=True) == [0, 1, 2, 3]

    # Test that names must exist.
    with pytest.raises(ValueError):
        vs.get_vars('e')

    # Test some queries.
    assert vs.get_vars('a') == [1]
    assert vs.get_vars('a', '*/b') == [1, 2]
    assert vs.get_vars('*/c', 'a') == [1, 3]
    assert vs.get_vars('*/c', '*/b', 'a') == [1, 2, 3]

    assert vs.get_vars('a', indices=True) == [0]
    assert vs.get_vars('a', '*/b', indices=True) == [0, 1]
    assert vs.get_vars('*/c', 'a', indices=True) == [0, 2]
    assert vs.get_vars('*/c', '*/b', 'a', indices=True) == [0, 1, 2]

    # Test some more queries.
    assert vs.get_vars('1/*') == [2]
    assert vs.get_vars('2/*') == [3, 4]
    assert vs.get_vars('1/*', '2/*') == [2, 3, 4]

    assert vs.get_vars('1/*', indices=True) == [1]
    assert vs.get_vars('2/*', indices=True) == [2, 3]
    assert vs.get_vars('1/*', '2/*', indices=True) == [1, 2, 3]

    # Test even more queries.
    assert vs.get_vars('*/b', '1/*') == [2]
    assert vs.get_vars('a', '2/*') == [1, 3, 4]
    assert vs.get_vars('a', '2/d', '2/*') == [1, 3, 4]
    assert vs.get_vars('2/d', '2/c', 'a', '1/*') == [1, 2, 3, 4]
    assert vs.get_vars('1/*') == [2]
    assert vs.get_vars('2/*') == [3, 4]
    assert vs.get_vars('1/*', '2/*') == [2, 3, 4]

    assert vs.get_vars('*/b', '1/*', indices=True) == [1]
    assert vs.get_vars('a', '2/*', indices=True) == [0, 2, 3]
    assert vs.get_vars('a', '2/d', '2/*', indices=True) == [0, 2, 3]
    assert vs.get_vars('2/d', '2/c', 'a', '1/*', indices=True) == [0, 1, 2, 3]
    assert vs.get_vars('1/*', indices=True) == [1]
    assert vs.get_vars('2/*', indices=True) == [2, 3]
    assert vs.get_vars('1/*', '2/*', indices=True) == [1, 2, 3]


def test_get_vars_cache_clearing():
    vs = Vars(np.float32)
    vs.get(name='var_a')
    assert vs.get_vars('var_*', indices=True) == [0]
    vs.get(name='var_b')
    assert vs.get_vars('var_*', indices=True) == [0, 1]


def test_get_set_vector():
    vs = Vars(np.float64)

    # Test stacking a matrix and a vector.
    vs.get(shape=(2,), name='a', init=np.array([1, 2]))
    vs.get(shape=(2, 2), name='b', init=np.array([[3, 4], [5, 6]]))
    allclose(vs.get_vector('a', 'b'), [1, 2, 3, 4, 5, 6])

    # Test setting elements.
    vs.set_vector(np.array([6, 5, 4, 3, 2, 1]), 'a', 'b')
    allclose(vs['a'], [6, 5])
    allclose(vs['b'], np.array([[4, 3], [2, 1]]))

    # Test setting elements in a differentiable way. This should allow for
    # any values.
    vs.set_vector(np.array(['1', '2', '3', '4', '5', '6']), 'a', 'b',
                  differentiable=True)
    assert np.all(vs['a'] == ['1', '2'])
    assert np.all(vs['b'] == np.array([['3', '4'], ['5', '6']]))


def test_get_and_init_tf():
    s = tf.Session()

    # Test `float32`.
    vs = Vars(tf.float32)
    a = vs.get(1., name='a')
    b = vs.get()
    assert len(vs.vars) == 2
    assert a.dtype.as_numpy_dtype == np.float32
    assert b.dtype.as_numpy_dtype == np.float32
    vs.init(s)
    assert s.run(vs['a']) == 1.

    # Test `float64`.
    vs = Vars(tf.float64)
    a = vs.get(1., name='a')
    b = vs.get()
    assert len(vs.vars) == 2
    assert a.dtype.as_numpy_dtype == np.float64
    assert b.dtype.as_numpy_dtype == np.float64
    vs.init(s)
    assert s.run(vs['a']) == 1.


def test_positive():
    vs = Vars(np.float64)
    for _ in range(10):
        assert vs.pos() >= 0


def test_bounded():
    vs = Vars(np.float64)
    for _ in range(10):
        v = vs.bnd(lower=10, upper=11)
        assert v >= 10
        assert v <= 11


def test_assignment():
    s = tf.Session()
    dispatch = Dispatcher()

    @dispatch(object)
    def convert(x):
        return x

    @dispatch(B.Torch)
    def convert(x):
        return x.numpy()

    @dispatch(B.TF)
    def convert(x):
        return s.run(x)

    for vs in [Vars(np.float64), Vars(tf.float64), Vars(torch.float64)]:
        # Generate some variables.
        vs.get(1., name='unbounded')
        vs.pos(2., name='positive')
        vs.bnd(3., lower=0, upper=10, name='bounded')

        if isinstance(vs.dtype, B.TFDType):
            vs.init(s)

        # Check that they have the right values.
        assert 1. == convert(vs['unbounded'])
        allclose(2., convert(vs['positive']))
        allclose(3., convert(vs['bounded']))

        # Assign some new values.
        convert(vs.assign('unbounded', 4.))
        convert(vs.assign('positive', 5.))
        convert(vs.assign('bounded', 6.))

        # Again check that they have the right values.
        assert 4. == convert(vs['unbounded'])
        allclose(5., convert(vs['positive']))
        allclose(6., convert(vs['bounded']))

        # Differentiably assign new values. This should allow for anything.
        vs.assign('unbounded', 'value', differentiable=True)
        assert vs['unbounded'] == 'value'

    s.close()


def test_detach_torch():
    vs = Vars(torch.float64)

    # Create a variable.
    vs.pos(1, name='a')

    # Initialise vector packer.
    vs.get_vector()

    # Make a detached copy.
    vs2 = vs.detach()

    # Require gradients for both.
    vs.requires_grad(True)
    vs2.requires_grad(True)

    # Do a backward pass.
    (vs2['a'] ** 2).backward()

    # Check that values are equal, but gradients only computed for one.
    assert vs['a'] == 1
    assert vs.get_vars('a')[0].grad is None
    assert vs2['a'] == 1
    assert vs2.get_vars('a')[0].grad == 2

    # Check that copied fields are, in fact, copies.
    del vs.transforms[:]
    del vs.inverse_transforms[:]
    vs.index_by_name.clear()
    assert len(vs2.transforms) > 0
    assert len(vs2.inverse_transforms) > 0
    assert len(vs2.index_by_name) > 0

    # Check that vector packer is copied.
    assert vs2.vector_packer != None


def test_requires_grad_detach_vars_torch():
    vs = Vars(torch.float64)
    vs.pos(1, name='a')

    # Test that gradients need to first be required.
    with pytest.raises(RuntimeError):
        (2 * vs['a']).backward()

    # Test that gradients can be required and are then computed.
    vs.requires_grad(True)
    (2 * vs['a']).backward()
    assert type(vs.vars[0].grad) != type(None)

    # Test that variables can be detached.
    vs.pos(1, name='b')
    result = 2 * vs['b']
    vs.detach_vars()
    with pytest.raises(RuntimeError):
        result.backward()


def test_source():
    vs = Vars(np.float32, source=np.ones(10))

    assert vs.get() == 1.
    approx(vs.pos(shape=(5,)), np.exp(np.ones(5)))
    approx(vs.pos(), np.exp(1.))
    with pytest.raises(ValueError):
        vs.pos(shape=(5,))

    # Test that the source variables are casted to the right data type.

    vs = Vars(np.float32, source=np.array([1]))
    assert vs.get().dtype == np.float32

    vs = Vars(np.float64, source=np.array([1]))
    assert vs.get().dtype == np.float64
