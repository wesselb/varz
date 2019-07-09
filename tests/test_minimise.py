# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np
import pytest
import tensorflow as tf
import torch
import lab as B

import varz.autograd
import varz.tensorflow
import varz.torch
from varz import Vars
from .util import approx, Value


@pytest.fixture(params=[(np.float64, varz.autograd.minimise_l_bfgs_b),
                        (torch.float64, varz.torch.minimise_l_bfgs_b),
                        (tf.float64, varz.tensorflow.minimise_l_bfgs_b)])
def dtype_minimise_l_bfgs_b(request):
    yield request.param


def test_optimise(dtype_minimise_l_bfgs_b):
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


def test_optimise_disconnected_gradient(dtype_minimise_l_bfgs_b):
    dtype, minimise = dtype_minimise_l_bfgs_b
    vs = Vars(dtype=dtype)
    vs.get(name='x')

    # Check that optimiser runs for objective that returns the constant zero.
    minimise(lambda v: B.cast(v.dtype, 0), vs)


def test_optimise_runtimeerror(dtype_minimise_l_bfgs_b):
    dtype, minimise = dtype_minimise_l_bfgs_b
    vs = Vars(dtype=dtype)

    first_call = Value(True)

    # Define an objective that sometimes fails after the first call.
    def f(vs_):
        if first_call.val:
            first_call.val = False
        else:
            if np.random.rand() > .5:
                raise RuntimeError('Fail!')
        return vs_.get(name='x', init=5.) ** 2

    # Check that the optimiser runs.
    minimise(f, vs)


def test_optimise_zero_calls(dtype_minimise_l_bfgs_b):
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
