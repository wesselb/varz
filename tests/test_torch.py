# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from lab import B
import torch

from . import Vars, minimise_l_bfgs_b

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam, allclose, approx


def test_optimise_torch():
    B.backend_to_torch()
    vs = Vars(dtype=torch.float64)

    # Initialise a variable that is not used.
    vs.get(name='other')

    # Define some objective.
    def f(vs_):
        return (-3 - vs_.pos(name='x', init=5.)) ** 2

    # Minimise it.
    val_opt = minimise_l_bfgs_b(f, vs, trace=False)

    # Check for equality up to five digits.
    yield approx, val_opt, 9, 5
    yield approx, vs['x'].detach().numpy(), 0, 5
