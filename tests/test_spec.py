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
        return vs_['z'], vs_.get(), vs_.pos(), vs_.bnd()

    assert f(1, vs, 2) == f(3, vs, 4)
    assert vs.names == names
    assert f(1, vs, 2) == f(3, vs, 4)
    assert vs.names == names
