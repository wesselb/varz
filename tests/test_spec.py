import pytest
import numpy as np
from varz.spec import _extract_prefix_and_f
from varz import (
    Vars,
    sequential,
    Unbounded,
    Positive,
    Bounded,
    parametrised
)

from .util import vs


def test_extract_prefix_and_f():
    def f():
        pass

    assert _extract_prefix_and_f(None) == ('', None)
    assert _extract_prefix_and_f(f) == ('', f)
    assert _extract_prefix_and_f('test') == ('test', None)


@pytest.mark.parametrize('decorator, names',
                         [(sequential, ['z', '0', '1', '2']),
                          (sequential(), ['z', '0', '1', '2']),
                          (sequential('x'), ['z', 'x0', 'x1', 'x2'])])
def test_sequential(vs, decorator, names):
    vs.get(0, name='z')

    @decorator
    def f(x, vs_, y):
        assert x == 1
        assert y == 2
        return vs_['z'], vs_.get(), vs_.pos(), vs_.bnd(lower=10, upper=11)

    # Test that the same variables are retrieved
    assert f(1, vs, 2) == f(1, vs, 2)
    assert vs.names == names
    assert f(1, vs, 2) == f(1, vs, 2)
    assert vs.names == names

    # Test correctness off variables.
    assert f(1, vs, 2)[0] == 0
    assert f(1, vs, 2)[2] >= 0
    assert 10 <= f(1, vs, 2)[3] <= 11


@pytest.mark.parametrize('decorator, types, names',
                         [(parametrised,
                           [Unbounded, Positive, Bounded(lower=10, upper=11)],
                           ['w', 'x', 'y', 'z']),
                          (parametrised(),
                           [Unbounded, Positive, Bounded(lower=10, upper=11)],
                           ['w', 'x', 'y', 'z']),
                          (parametrised('var_'),
                           [Unbounded, Positive, Bounded(lower=10, upper=11)],
                           ['w', 'var_x', 'var_y', 'var_z'])])
def test_parametrised(vs, decorator, types, names):
    vs.get(0, name='w')

    @decorator
    def f(a, x: types[0], vs_, y: types[1], b, z: types[2], c=None):
        assert a == 1
        assert b == 2
        assert c is None
        return vs_['w'], x, y, z

    # Test that the same variables are retrieved
    assert f(1, vs, 2) == f(1, vs, 2)
    assert vs.names == names
    assert f(1, vs, 2) == f(1, vs, 2)
    assert vs.names == names

    # Test correctness off variables.
    assert f(1, vs, 2)[0] == 0
    assert f(1, vs, 2)[2] >= 0
    assert 10 <= f(1, vs, 2)[3] <= 11


def test_parametrised_arguments(vs):
    @parametrised
    def f(vs_, x: Unbounded, a, y: Unbounded, b=2, z: Unbounded = 10, c=3):
        return a, b, c

    with pytest.raises(ValueError):
        f(vs)

    assert f(vs, 1) == (1, 2, 3)
    assert f(vs, a=1) == (1, 2, 3)

    assert f(vs, 1, 4) == (1, 4, 3)
    assert f(vs, 1, b=4) == (1, 4, 3)
    assert f(vs, a=1, b=4) == (1, 4, 3)

    assert f(vs, 1, 4, 5) == (1, 4, 5)
    assert f(vs, 1, 4, c=5) == (1, 4, 5)
    assert f(vs, a=1, b=4, c=5) == (1, 4, 5)

    # Test unspecified positional argument.
    with pytest.raises(ValueError):
        f(vs, 1, 2, 3, 4)

    # Test unspecified keyword argument.
    with pytest.raises(ValueError):
        f(vs, 1, 2, d=3)


def test_parametrised_variable_container(vs):
    @parametrised
    def f(x, y, z: Unbounded):
        pass

    # Test that a variable container must be present.
    with pytest.raises(ValueError):
        f(1, 2)

    # Test that only one variable container can be present.
    with pytest.raises(ValueError):
        f(vs, vs)


def test_parametrised_double_initial_value(vs):
    @parametrised
    def f(vs_, x: Unbounded(init=5) = 5):
        pass

    with pytest.raises(ValueError):
        f(vs)


def test_parametrised_double_name(vs):
    @parametrised
    def f(vs_, x: Unbounded(name='x')):
        pass

    with pytest.raises(ValueError):
        f(vs)
