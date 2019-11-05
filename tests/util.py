import lab as B
from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher

__all__ = ['allclose', 'approx', 'Value']

_dispatch = Dispatcher()


class Value:
    def __init__(self, val):
        self.val = val


@_dispatch({B.Number, B.NP, list})
def _to_numpy(x):
    return x


@_dispatch({B.Torch, B.TF})
def _to_numpy(x):
    return x.numpy()


def approx(x, y, digits=7):
    x = _to_numpy(x)
    y = _to_numpy(y)
    assert_array_almost_equal(x, y, decimal=digits)


def allclose(x, y):
    x = _to_numpy(x)
    y = _to_numpy(y)
    assert_allclose(x, y)
