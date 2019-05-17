# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import lab.torch as B
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

__all__ = ['minimise_l_bfgs_b']

log = logging.getLogger(__name__)


def minimise_l_bfgs_b(f,
                      vs,
                      f_calls=10000,
                      iters=1000,
                      trace=False,
                      names=None):
    """Minimise a function with L-BFGS-B in PyTorch.

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
    zero = B.cast(vs.dtype, 0)

    # Run function once to ensure that all variables are initialised and
    # available.
    val_init = f(vs)

    # SciPy doesn't perform zero iterations, so handle that edge case manually.
    if iters == 0 or f_calls == 0:
        return val_init

    # Detach variables from the current computation graph.
    vs.detach_vars()

    # Extract initial value.
    x0 = vs.get_vector(*names).numpy()

    # Turn on gradient computation.
    vs.requires_grad(False)
    vs.requires_grad(True, *names)

    def f_wrapped(x):
        # Update variable manager.
        vs.set_vector(B.cast(vs.dtype, x), *names)

        # Compute objective function value, detach, and convert to NumPy.
        try:
            obj_value = f(vs)
            obj_value.backward()
            obj_value = obj_value.detach_().numpy()
        except RuntimeError as e:
            log.warning('Caught exception during function evaluation: '
                        '"{}". Returning NaN.'.format(e))
            obj_value = np.nan

        # Loop over variable manager to extract gradients and zero them.
        grads = []
        for var in vs.get_vars(*names):
            # Save gradient if there is one.
            if var.grad is None:
                grads.append(zero)
            else:
                grads.append(var.grad.clone())
                var.grad.data.zero_()  # Clear gradient.

        # Stack, detach, and convert to NumPy.
        grad = vs.vector_packer.pack(*grads).detach_().numpy()

        return obj_value, grad

    # Perform optimisation routine.
    x_opt, val_opt, info = fmin_l_bfgs_b(func=f_wrapped,
                                         x0=x0,
                                         maxiter=iters,
                                         maxfun=f_calls,
                                         callback=None,
                                         disp=1 if trace else 0)

    # TODO: Print some report if `trace` is `True`.
    pass

    # Turn off gradient computation again.
    vs.requires_grad(False)

    # Return optimal value.
    return val_opt
