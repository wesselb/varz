import lab as B
import numpy as np
import pytest
import tensorflow as tf
import torch
import wbml.out
from numpy.testing import assert_allclose
from plum import Dispatcher
from varz import Vars
from wbml import out as out

__all__ = [
    "Value",
    "approx",
    # Numerical checks:
    "assert_lower_triangular",
    "assert_positive_definite",
    "assert_orthogonal",
    # Fixtures:
    "dtype",
    "vs",
    "vs_source",
    # Mocks:
    "KV",
    "OutStream",
]

_dispatch = Dispatcher()


class Value:
    def __init__(self, val):
        self.val = val


def approx(x, y, atol=1e-12, rtol=1e-8):
    assert_allclose(*B.to_numpy(x, y), atol=atol, rtol=rtol)


# Numerical checks:


def assert_lower_triangular(x):
    """Assert that a matrix is lower triangular."""
    # Check that matrix is square.
    assert B.shape(x)[0] == B.shape(x)[1]

    # Check that upper part is all zeros.
    upper = x[np.triu_indices(B.shape(x)[0], k=1)]
    approx(upper, B.zeros(upper))


def assert_positive_definite(x):
    """Assert that a matrix is positive definite."""
    # Check that Cholesky decomposition succeeds.
    B.cholesky(x)


def assert_orthogonal(x):
    """Assert that a matrix is orthogonal."""
    # Check that matrix is square.
    assert B.shape(x)[0] == B.shape(x)[1]

    # Check that its transpose is its inverse.
    approx(B.matmul(x, x, tr_a=True), B.eye(x))


# Fixtures:


@pytest.fixture(params=[np.float64, torch.float64, tf.float64])
def dtype(request):
    return request.param


@pytest.fixture()
def vs():
    return Vars(np.float64)


@pytest.fixture(params=[False, True])
def vs_source(request):
    if request.param:
        source = B.randn(np.float64, 1000)
        return Vars(np.float64, source=source)
    else:
        return Vars(np.float64)


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
        self.output = ""

    def write(self, msg):
        self.output += msg

    def __enter__(self):
        self._orig_streams = list(out.streams)
        out.streams[:] = [self]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        out.streams[:] = self._orig_streams
