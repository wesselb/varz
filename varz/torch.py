# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

from lab import B
from scipy.optimize import fmin_l_bfgs_b

__all__ = ['minimise_l_bfgs_b']

log = logging.getLogger(__name__)


def minimise_l_bfgs_b(f,
                      vs,
                      f_calls=1000,
                      iters=10_000,
                      trace=True,
                      names=None,
                      groups=None):
    """Minimise a function with L-BFGS-B.

    Args:
        f (function): Function to optimise.
        vs (:class:`.vars.Vars`): Variable manager.
        f_calls (int, optional): Maximum number of function calls. Defaults to
            `1000`.
        iters (int, optional): Maximum number of iterations. Defaults to
            `10_000`.
        trace (bool, optional): Show trace of optimisation. Defaults to `False`.
        names (list, optional): List of names of variables to optimise.
        groups (list, optional): List of groups of variables to optimise.

    Returns:
        float: Final objective function value.
    """
    names = [] if names is None else names
    zero = B.cast(0, dtype=vs.dtype)

    def f_wrapped(x):
        # Update variable manager.
        vs.set_vector(B.cast(x, dtype=vs.dtype), *names, groups=groups)

        # Compute objective function value, detach, and convert to NumPy.
        obj_value = f(vs)
        obj_value.backward()
        obj_value = obj_value.detach().numpy()

        # Loop over variable manager to extract gradients and zero them.
        grads = []
        for var in vs.get_vars(*names, groups=groups):
            # Save gradient if there is one.
            if var.grad is None:
                grads.append(zero)
            else:
                grads.append(var.grad.clone())
                var.grad.data.zero_()  # Clear gradient.

        # Stack, detach, and convert to NumPy.
        grad = vs.vector_packer.pack(*grads).detach().numpy()

        return obj_value, grad

    # Run function once to ensure that all variables are initialised and
    # available.
    val_init = f(vs)

    # Extract initial value.
    x0 = vs.get_vector(*names, groups=groups).detach().numpy()

    # Perform optimisation routine.
    x_opt, val_opt, info = fmin_l_bfgs_b(func=f_wrapped,
                                         x0=x0,
                                         maxiter=iters,
                                         maxfun=f_calls,
                                         callback=None,
                                         disp=1 if trace else 0)

    # TODO: Print some report if `trace` is `True`.

    # Return optimal value.
    return val_opt
