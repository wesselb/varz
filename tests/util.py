import lab as B
import numpy as np
import pytest
import tensorflow as tf
import torch
import wbml.out
from numpy.testing import assert_allclose, assert_array_almost_equal
from plum import Dispatcher
from wbml import out as out
from varz import Vars

__all__ = ['Value',
           'allclose',
           'approx',

           # Fixtures:
           'dtype',
           'vs',

           # Mocks:
           'KV',
           'OutStream']

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


# Fixtures:

@pytest.fixture(params=[np.float64, torch.float64, tf.float64])
def dtype(request):
    yield request.param


@pytest.fixture()
def vs():
    vs = Vars(np.float64)
    yield vs


# Mocks:


class KV:
    """Mock `wbml.out.kv`."""

    def __init__(self):
        self.keys = []
        self.values = []
        self._kv = None

    def __call__(self, key, value):
        self.keys.append(key)
        self.values.append(value)

    def __enter__(self):
        self._kv = wbml.out.kv
        wbml.out.kv = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        wbml.out.kv = self._kv


class OutStream:
    """Mock the streams of `wbml.out`."""

    def __init__(self):
        self.output = ''

    def write(self, msg):
        self.output += msg

    def __enter__(self):
        self._orig_streams = list(out.streams)
        out.streams[:] = [self]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        out.streams[:] = self._orig_streams
