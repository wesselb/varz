import logging
import traceback
from functools import wraps

from .adam import ADAM

import lab as B
import numpy as np
import wbml.out as out
from plum import convert
from scipy.optimize import fmin_l_bfgs_b

__all__ = ["minimise_l_bfgs_b", "minimise_adam"]

log = logging.getLogger(__name__)


def minimise_l_bfgs_b(
    f, vs, f_calls=10000, iters=1000, trace=False, names=None, jit=False
):  # pragma: no cover
    """Minimise a function with L-BFGS-B.

    Args:
        f (function): Function to optimise.
        vs (:class:`.vars.Vars`): Variable manager.
        f_calls (int, optional): Maximum number of function calls. Defaults to `10000`.
        iters (int, optional): Maximum number of iterations. Defaults to `1000`.
        trace (bool, optional): Show trace of optimisation. Defaults to `False`.
        names (list, optional): List of names of variables to optimise.
            Defaults to all variables.
        jit (bool, optional): Use a JIT if one is available. Defaults to `False`.

    Returns:
        float: Final objective function value.
    """
    raise RuntimeError("Call a backend-specific optimiser instead.")


def minimise_adam(
    f,
    vs,
    iters=1000,
    rate=1e-3,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    local_rates=True,
    trace=False,
    names=None,
    jit=False,
):  # pragma: no cover
    """Minimise a function with Adam.

    Args:
        f (function): Function to optimise.
        vs (:class:`.vars.Vars`): Variable manager.
        iters (int, optional): Maximum number of iterations. Defaults to `1000`.
        rate (float, optional): Learning rate. Defaults to `1e-3`.
        beta1 (float, optional): Exponential decay for mean. Defaults to `0.9`.
        beta2 (float, optional): Exponential decay for second moment. Defaults to
            `0.999`.
        epsilon (float, optional): Small value to prevent division by zero.
            Defaults to `1e-8`.
        local_rates (bool, optional): Use local learning rates. Set to `False` to
            use one global learning rate. Defaults to `True`.
        trace (bool, optional): Show trace of optimisation. Defaults to `False`.
        names (list, optional): List of names of variables to optimise.
            Defaults to all variables.
        jit (bool, optional): Use a JIT if one is available. Defaults to `False`.

    Returns:
        float: Final objective function value.
    """
    raise RuntimeError("Call a backend-specific optimiser instead.")


def make_l_bfgs_b(wrap_f):
    """Create `minimise_l_bfgs_b` given a function wrapper.

    Args:
        wrap_f (function): Function wrapper.

    Returns:
        function: `minimise_l_bfgs_b`.
    """

    @wraps(minimise_l_bfgs_b)
    def _minimise_l_bfgs_b(
        f, vs, f_calls=10000, iters=1000, trace=False, names=None, jit=False
    ):
        names = [] if names is None else names

        # Run function once to ensure that all variables are initialised and
        # available.
        val_init = f(vs)

        # SciPy doesn't perform zero iterations, so handle that edge case
        # manually.
        if iters == 0 or f_calls == 0:
            return B.to_numpy(val_init)

        # Extract initial value.
        x0 = B.to_numpy(vs.get_vector(*names))

        # Wrap the function and get the list of function evaluations.
        f_vals, f_wrapped = wrap_f(vs, names, f, jit)

        # Perform optimisation routine.
        def perform_minimisation(callback_=lambda _: None):
            return fmin_l_bfgs_b(
                func=f_wrapped,
                x0=x0,
                maxiter=iters,
                maxfun=f_calls,
                callback=callback_,
                disp=0,
            )

        if trace:
            # Print progress during minimisation.
            with out.Progress(
                name='Minimisation of "{}"'.format(f.__name__), total=iters
            ) as progress:

                def callback(_):
                    progress({"Objective value": np.min(f_vals)})

                x_opt, val_opt, info = perform_minimisation(callback)

            with out.Section("Termination message"):
                out.out(convert(info["task"], str))
        else:
            # Don't print progress; simply perform minimisation.
            x_opt, val_opt, info = perform_minimisation()

        vs.set_vector(x_opt, *names)  # Assign optimum.

        return val_opt  # Return optimal value.

    return _minimise_l_bfgs_b


def make_adam(wrap_f):
    """Create `minimise_adam` given a function wrapper.

    Args:
        wrap_f (function): Function wrapper.

    Returns:
        function: `minimise_adam`.
    """

    @wraps(minimise_adam)
    def _minimise_adam(
        f,
        vs,
        iters=1000,
        rate=1e-3,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        local_rates=True,
        trace=False,
        names=None,
        jit=False,
    ):
        names = [] if names is None else names

        # Run function once to ensure that all variables are initialised and
        # available.
        val_init = f(vs)

        # Handle the edge case of zero iterations.
        if iters == 0:
            return B.to_numpy(val_init)

        # Extract initial value.
        x0 = B.to_numpy(vs.get_vector(*names))

        # Wrap the function.
        _, f_wrapped = wrap_f(vs, names, f, jit)

        def perform_minimisation(callback_=lambda _: None):
            # Perform optimisation routine.
            x = x0
            obj_value = None
            adam = ADAM(
                rate=rate,
                beta1=beta1,
                beta2=beta2,
                epsilon=epsilon,
                local_rates=local_rates,
            )

            for i in range(iters):
                obj_value, grad = f_wrapped(x)
                callback_(obj_value)
                x = adam.step(x, grad)

            return x, obj_value

        if trace:
            # Print progress during minimisation.
            with out.Progress(
                name='Minimisation of "{}"'.format(f.__name__), total=iters
            ) as progress:

                def callback(obj_value):
                    progress({"Objective value": obj_value})

                x_opt, obj_value = perform_minimisation(callback)
        else:
            x_opt, obj_value = perform_minimisation()

        vs.set_vector(x_opt, *names)  # Assign optimum.

        return obj_value  # Return last objective value.

    return _minimise_adam


def exception(x, e):
    """In the case that an exception is raised during function evaluation,
    print a warning and return NaN for the function value and gradient.

    Args:
        x (tensor): Current input.
        e (:class:`Exception`): Caught exception.

    Returns:
        tuple: Tuple containing NaN and NaNs for the gradient.
    """
    with out.Section("Caught exception during function evaluation"):
        out.out(traceback.format_exc().strip())
    grad_nan = np.empty(x.shape)
    grad_nan[:] = np.nan
    return np.nan, grad_nan
