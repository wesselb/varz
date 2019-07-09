# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from autograd import value_and_grad

from ..minimise import make_l_bfgs_b

__all__ = ['minimise_l_bfgs_b']

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f):
    # Differentiable assignments will overwrite the variables, so make a copy.
    vs_copy = vs.copy()

    def f_vectorised(x):
        vs_copy.set_vector(x, *names, differentiable=True)
        return f(vs_copy)

    def f_wrapped(x):
        # Compute objective function value.
        try:
            return value_and_grad(f_vectorised)(x)
        except RuntimeError as e:
            log.warning('Caught exception during function evaluation: '
                        '"{}". Returning NaN.'.format(e))
            grad_nan = np.empty(x.shape)
            grad_nan[:] = np.nan
            return np.nan, grad_nan

    return f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
