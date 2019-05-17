# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from autograd import value_and_grad
from scipy.optimize import fmin_l_bfgs_b

__all__ = ['minimise_l_bfgs_b']

log = logging.getLogger(__name__)


def minimise_l_bfgs_b(f,
                      vs,
                      f_calls=10000,
                      iters=1000,
                      trace=False,
                      names=None):
    """Minimise a function with L-BFGS-B in NumPy and AutoGrad.

    Args:
        f (function): Function to optimise.
        vs (:class:`.vars.Vars`): Variable manager.
        f_calls (int, optional): Maximum number of function calls. Defaults to
            `10000`.
        iters (int, optional): Maximum number of iterations. Defaults to
            `1000`.
        trace (bool, optional): Show trace of optimisation. Defaults to `False`.
        names (list, optional): List of names of variables to optimise. Defaults
            to all variables.

    Returns:
        float: Final objective function value.
    """
    names = [] if names is None else names

    # Run function once to ensure that all variables are initialised and
    # available.
    val_init = f(vs)

    # SciPy doesn't perform zero iterations, so handle that edge case manually.
    if iters == 0 or f_calls == 0:
        return val_init

    # Extract initial value.
    x0 = vs.get_vector(*names)

    # Wrap `f` to have it take in vector, for compatibility with AutoGrad.
    def f_vectorised(x):
        vs.set_vector(x, *names, differentiable=True)
        return f(vs)

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

    # Perform optimisation routine.
    x_opt, val_opt, info = fmin_l_bfgs_b(func=f_wrapped,
                                         x0=x0,
                                         maxiter=iters,
                                         maxfun=f_calls,
                                         callback=None,
                                         disp=1 if trace else 0)

    # Due to differentiable assignments, the variables are now AutoGrad
    # objects. Convert them back.
    for i in vs.get_vars(*names, indices=True):
        vs.vars[i] = vs.vars[i]._value

    # TODO: Print some report if `trace` is `True`.
    pass

    # Return optimal value.
    return val_opt
