import numpy as np
import pytest
import torch
from varz import Vars
import lab as B

# noinspection PyUnresolvedReferences
from .util import (
    allclose,
    approx,
    KV,
    dtype,
    vs,
    vs_source,
    assert_lower_triangular,
    assert_positive_definite,
    assert_orthogonal
)


def test_get_vars():
    vs = Vars(np.int)

    # This test also tests that `Vars.get_vars` always returns the collection
    # of variables in the right order. This is important for optimisation.

    # Initialise some variables.
    vs.get(1, name='a')
    vs.get(2, name='1/b')
    vs.get(3, name='2/c')
    vs.unbounded(4, name='2/d')  # Test alias.

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


def test_get_vars_cache_clearing(vs):
    vs.get(name='var_a')
    assert vs.get_vars('var_*', indices=True) == [0]
    vs.get(name='var_b')
    assert vs.get_vars('var_*', indices=True) == [0, 1]


def test_unbounded(vs_source):
    for _ in range(10):
        vs_source.get()
        vs_source.unbounded()


def test_unbounded_init(vs):
    vs.get(1, name='x')
    allclose(vs['x'], 1)


def test_unbounded_assignment(vs):
    vs.get(1, name='x')
    vs.assign('x', 2)
    allclose(vs['x'], 2)


def test_positive(vs_source):
    for _ in range(10):
        assert vs_source.pos() >= 0
        assert vs_source.positive() >= 0


def test_positive_init(vs):
    vs.pos(1, name='x')
    allclose(vs['x'], 1)


def test_positive_assignment(vs):
    vs.pos(1, name='x')
    vs.assign('x', 2)
    allclose(vs['x'], 2)


def test_bounded(vs_source):
    for _ in range(10):
        assert 10 <= vs_source.bnd(lower=10, upper=11) <= 11
        assert 10 <= vs_source.bounded(lower=10, upper=11) <= 11


def test_bounded_init(vs):
    vs.bnd(2, name='x', lower=1, upper=4)
    allclose(vs['x'], 2)


def test_bounded_assignment(vs):
    vs.bnd(2, name='x', lower=1, upper=4)
    vs.assign('x', 3)
    allclose(vs['x'], 3)


def test_lower_triangular(vs_source):
    for _ in range(10):
        assert B.shape(vs_source.tril(shape=(5, 5))) == (5, 5)
        assert_lower_triangular(vs_source.tril(shape=(5, 5)))
        assert_lower_triangular(vs_source.lower_triangular(shape=(5, 5)))


def test_lower_triangular_init(vs):
    x = vs.tril(shape=(5, 5))

    vs.tril(x, name='x')
    allclose(vs['x'], x)

    with pytest.raises(ValueError):
        vs.tril()
    with pytest.raises(ValueError):
        vs.tril(shape=5)
    with pytest.raises(ValueError):
        vs.tril(shape=(5, 6))


def test_lower_triangular_assignment(vs):
    x = vs.tril(shape=(5, 5))

    vs.tril(shape=(5, 5), name='x')
    vs.assign('x', x)
    allclose(vs['x'], x)


def test_positive_definite(vs_source):
    for _ in range(10):
        assert B.shape(vs_source.pd(shape=(5, 5))) == (5, 5)
        assert_positive_definite(vs_source.pd(shape=(5, 5)))
        assert_positive_definite(vs_source.positive_definite(shape=(5, 5)))


def test_positive_definite_init(vs):
    x = vs.pd(shape=(5, 5))

    vs.pd(x, name='x')
    allclose(vs['x'], x)

    with pytest.raises(ValueError):
        vs.pd()
    with pytest.raises(ValueError):
        vs.pd(shape=5)
    with pytest.raises(ValueError):
        vs.pd(shape=(5, 6))


def test_positive_definite_assignment(vs):
    x = vs.pd(shape=(5, 5))

    vs.pd(shape=(5, 5), name='x')
    vs.assign('x', x)
    allclose(vs['x'], x)


def test_orthogonal(vs_source):
    for i in range(10):
        assert B.shape(vs_source.orth(shape=(5, 5))) == (5, 5)
        assert_orthogonal(vs_source.orth(shape=(5, 5)))
        assert_orthogonal(vs_source.orthogonal(shape=(5, 5)))


def test_orthogonal_init(vs):
    x = vs.orth(shape=(5, 5))

    vs.orth(x, name='x')
    allclose(vs['x'], x)

    with pytest.raises(ValueError):
        vs.orth()
    with pytest.raises(ValueError):
        vs.orth(shape=5)
    with pytest.raises(ValueError):
        vs.orth(shape=(5, 6))


def test_orthogonal_assignment(vs):
    x = vs.orth(shape=(5, 5))

    vs.orth(shape=(5, 5), name='x')
    vs.assign('x', x)
    allclose(vs['x'], x)


def test_get_set_vector(dtype):
    vs = Vars(dtype=dtype)

    # Test stacking a matrix and a vector.
    vs.get(shape=(2,), name='a', init=np.array([1, 2]))
    vs.get(shape=(2, 2), name='b', init=np.array([[3, 4], [5, 6]]))
    allclose(vs.get_vector('a', 'b'), [1, 2, 3, 4, 5, 6])

    # Test setting elements.
    vs.set_vector(np.array([6, 5, 4, 3, 2, 1]), 'a', 'b')
    allclose(vs['a'], np.array([6, 5]))
    allclose(vs['b'], np.array([[4, 3], [2, 1]]))

    # Test setting elements in a differentiable way. This should allow for
    # any values.
    vs.set_vector(np.array(['1', '2', '3', '4', '5', '6']), 'a', 'b',
                  differentiable=True)
    assert np.all(vs['a'] == ['1', '2'])
    assert np.all(vs['b'] == np.array([['3', '4'], ['5', '6']]))


def test_differentiable_assignment(vs):
    # Differentiably assigning new values should allow for anything.
    vs.get(name='x')
    vs.assign('x', 'value', differentiable=True)
    assert vs['x'] == 'value'


def test_copy_torch():
    vs = Vars(torch.float64)

    # Create a variable.
    vs.pos(1, name='a')

    # Initialise vector packer.
    vs.get_vector()

    # Make a normal and detached copy.
    vs_copy = vs.copy()
    vs_detached = vs.copy(detach=True)

    # Require gradients for both.
    vs.requires_grad(True)
    vs_detached.requires_grad(True)

    # Do a backward pass.
    (vs_detached['a'] ** 2).backward()

    # Check that values are equal, but gradients only computed for one.
    assert vs['a'] == 1
    assert vs.get_vars('a')[0].grad is None
    assert vs_detached['a'] == 1
    assert vs_detached.get_vars('a')[0].grad == 2

    # Check that copied fields are, in fact, copies, and that the vector packer
    # is also copied.
    del vs.transforms[:]
    del vs.inverse_transforms[:]
    del vs.vars[:]
    vs.name_to_index.clear()

    for vs_copied in [vs_copy, vs_detached]:
        assert len(vs_copied.transforms) > 0
        assert len(vs_copied.inverse_transforms) > 0
        assert len(vs_copied.vars) > 0
        assert len(vs_copied.name_to_index) > 0
        assert vs_copied.vector_packer != None


def test_requires_grad_detach_vars_torch():
    vs = Vars(torch.float64)
    vs.pos(1, name='a')

    # Test that gradients need to first be required.
    with pytest.raises(RuntimeError):
        (2 * vs['a']).backward()

    # Test that gradients can be required and are then computed.
    vs.requires_grad(True)
    (2 * vs['a']).backward()
    assert vs.vars[0].grad is not None

    # Test that variables can be detached.
    vs.pos(1, name='b')
    result = 2 * vs['b']
    vs.detach()
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


def test_names(vs):
    assert vs.names == []
    vs.get()
    assert vs.names == []
    vs.pos(name='a')
    assert vs.names == ['a']
    vs.bnd(name='b')
    assert vs.names == ['a', 'b']
    vs.get(name='c')
    assert vs.names == ['a', 'b', 'c']


def test_print(vs):
    with KV() as mock:
        vs.print()
        assert mock.keys == []
        assert mock.values == []

        vs.get()
        vs.print()
        assert mock.keys == []
        assert mock.values == []

        vs.print()
        vs.get(1, name='a')
        vs.print()
        assert mock.keys == ['a']
        assert mock.values == [1]

        vs.get(2, name='b')
        vs.print()
        assert mock.keys == ['a', 'a', 'b']
        assert mock.values == [1, 1, 2]

        vs.get(3, name='c')
        vs.print()
        assert mock.keys == ['a', 'a', 'b', 'a', 'b', 'c']
        assert mock.values == [1, 1, 2, 1, 2, 3]
