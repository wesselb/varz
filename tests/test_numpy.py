# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import numpy as np

from . import Vars
from varz.numpy import minimise_l_bfgs_b
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose, approx


def test_optimise():
    vs = Vars(dtype=np.float64)

    # Initialise a variable that is not used.
    vs.get(name='other')

    # Define some objective.
    def f(vs_):
        return (-3 - vs_.pos(name='x', init=5.)) ** 2

    # Minimise it.
    val_opt = minimise_l_bfgs_b(f, vs)

    # Check for equality up to five digits.
    yield approx, val_opt, 9, 5
    yield approx, vs['x'], 0, 5


first_call = True


def test_optimise_runtimeerror():
    vs = Vars(dtype=np.float64)

    # Define an objective that sometimes fails after the first call.
    def f(vs_):
        global first_call
        if first_call:
            first_call = False
        else:
            if np.random.rand() > .5:
                raise RuntimeError('Fail!')
        return vs_.get(name='x', init=5.) ** 2

    # Check that the optimiser runs.
    yield minimise_l_bfgs_b, f, vs


calls = 0


def test_optimise_zero_calls():
    vs = Vars(dtype=np.float64)

    def f(vs_):
        global calls
        calls += 1
        return vs_.get(name='x', init=5.) ** 2

    # Check that running the optimiser for zero iterations only incurs a
    # single call.
    minimise_l_bfgs_b(f, vs, iters=0)
    yield eq, calls, 1
    minimise_l_bfgs_b(f, vs, f_calls=0)
    yield eq, calls, 2
