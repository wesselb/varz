# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import lab as B
import numpy as np
import pytest
import tensorflow as tf
import torch
import varz.autograd
import varz.tensorflow
import varz.torch
import wbml.out as out
from varz import Vars

from .util import approx, Value


class OutStream(object):
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


@pytest.fixture(params=[(np.float64, varz.autograd.minimise_l_bfgs_b),
                        (torch.float64, varz.torch.minimise_l_bfgs_b),
                        (tf.float64, varz.tensorflow.minimise_l_bfgs_b)])
def dtype_minimise_l_bfgs_b(request):
    yield request.param


def test_minimise(dtype_minimise_l_bfgs_b):
    dtype, minimise = dtype_minimise_l_bfgs_b
    vs = Vars(dtype=dtype)

    # Initialise a variable that is not used.
    vs.get(name='other')

    # Define some objective.
    def f(vs_):
        return (-3 - vs_.pos(name='x', init=5.)) ** 2

    # Minimise it.
    val_opt = minimise(f, vs)

    # Check for equality up to five digits.
    approx(val_opt, 9, digits=5)
    approx(vs['x'], 0, digits=5)


def test_minimise_disconnected_gradient(dtype_minimise_l_bfgs_b):
    dtype, minimise = dtype_minimise_l_bfgs_b
    vs = Vars(dtype=dtype)
    vs.get(name='x')

    # Check that optimiser runs for objective that returns the constant zero.
    minimise(lambda v: B.cast(v.dtype, 0), vs)


def test_minimise_exception(dtype_minimise_l_bfgs_b):
    dtype, minimise = dtype_minimise_l_bfgs_b
    vs = Vars(dtype=dtype)

    first_call = Value(True)

    # Define an objective that sometimes fails after the first call.
    def f(vs_):
        if first_call.val:
            first_call.val = False
        else:
            if np.random.rand() > .5:
                raise Exception('Fail!')
        return vs_.get(name='x', init=5.) ** 2

    # Check that the optimiser runs.
    minimise(f, vs)


def test_minimise_zero_calls(dtype_minimise_l_bfgs_b):
    dtype, minimise = dtype_minimise_l_bfgs_b
    vs = Vars(dtype=dtype)

    calls = Value(0)

    def f(vs_):
        calls.val += 1
        return vs_.get(name='x', init=5.) ** 2

    # Check that running the optimiser for zero iterations only incurs a
    # single call.
    minimise(f, vs, iters=0)
    assert calls.val == 1
    minimise(f, vs, f_calls=0)
    assert calls.val == 2


def test_minimise_trace(dtype_minimise_l_bfgs_b):
    dtype, minimise = dtype_minimise_l_bfgs_b

    def f(vs_):
        return vs_.get(name='x') ** 2

    # Test that `trace=False` prints nothing.
    with OutStream() as stream:
        minimise(f, Vars(dtype=dtype), trace=False)
        assert stream.output == ''

    # Test that `trace=False` prints something.
    with OutStream() as stream:
        minimise(f, Vars(dtype=dtype), trace=True)
        assert stream.output != ''
