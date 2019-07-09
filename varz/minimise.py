# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import lab as B
from plum import Dispatcher
from scipy.optimize import fmin_l_bfgs_b
import numpy as np

__all__ = ['minimise_l_bfgs_b']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


@_dispatch(B.NP)
def _to_numpy(x):
    return x


@_dispatch({B.TF, B.Torch})
def _to_numpy(x):
    return x.numpy()


def minimise_l_bfgs_b(f,
                      vs,
                      f_calls=10000,
                      iters=1000,
                      trace=False,
                      names=None):  # pragma: no cover
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
    raise RuntimeError('Call a backend-specific optimiser instead.')


def make_l_bfgs_b(wrap_f):
    """Create `minimise_l_bfgs_b` given a function wrapper.

    Args:
        wrap_f (function): Function wrapper.

    Returns:
        function: `minimise_l_bfgs_b`.
    """
    def minimise_l_bfgs_b(f,
                          vs,
                          f_calls=10000,
                          iters=1000,
                          trace=False,
                          names=None):
        names = [] if names is None else names

        # Run function once to ensure that all variables are initialised and
        # available.
        val_init = f(vs)

        # SciPy doesn't perform zero iterations, so handle that edge case
        # manually.
        if iters == 0 or f_calls == 0:
            return _to_numpy(val_init)

        # Extract initial value.
        x0 = _to_numpy(vs.get_vector(*names))

        # Perform optimisation routine.
        x_opt, val_opt, info = fmin_l_bfgs_b(func=wrap_f(vs, names, f),
                                             x0=x0,
                                             maxiter=iters,
                                             maxfun=f_calls,
                                             callback=None,
                                             disp=1 if trace else 0)
        vs.set_vector(x_opt, *names)  # Assign optimum.

        # Return optimal value.
        return val_opt

    return minimise_l_bfgs_b


def exception(x, e):
    """In the case that an exception is raised during function evaluation,
    print a warning and return NaN for the function value and gradient.

    Args:
        x (tensor): Current input.
        e (:class:`Exception`): Caught exception.

    Returns:
        tuple: Tuple containing NaN and NaNs for the gradient.
    """
    log.warning('Caught exception during function evaluation: '
                '"{}". Returning NaN.'.format(e))
    grad_nan = np.empty(x.shape)
    grad_nan[:] = np.nan
    return np.nan, grad_nan
