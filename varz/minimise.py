import traceback
from functools import wraps
import importlib

from .adam import ADAM

import lab as B
import numpy as np
import wbml.out as out
from plum import Dispatcher
from plum import convert, List
from scipy.optimize import fmin_l_bfgs_b

__all__ = ["minimise_l_bfgs_b", "minimise_adam"]

_dispatch = Dispatcher()


@_dispatch
def _get_minimise_f(dtype: B.NPDType, name: str):
    mod = importlib.import_module("varz.autograd.minimise")
    return getattr(mod, name)


@_dispatch
def _get_minimise_f(dtype: B.TFDType, name: str):
    mod = importlib.import_module("varz.tensorflow.minimise")
    return getattr(mod, name)


@_dispatch
def _get_minimise_f(dtype: B.TorchDType, name: str):
    mod = importlib.import_module("varz.torch.minimise")
    return getattr(mod, name)


@_dispatch
def _get_minimise_f(dtype: B.JAXDType, name: str):
    mod = importlib.import_module("varz.jax.minimise")
    return getattr(mod, name)


def _convert_and_validate_names(names):
    if names is None:
        names = []
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, List[str]):
        raise ValueError("Keyword `names` must be a list of strings.")
    return names


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
    minimise_f = _get_minimise_f(convert(vs, tuple)[0].dtype, "minimise_l_bfgs_b")
    return minimise_f(
        f=f,
        vs=vs,
        f_calls=f_calls,
        iters=iters,
        trace=trace,
        names=names,
        jit=jit,
    )


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
    minimise_f = _get_minimise_f(convert(vs, tuple)[0].dtype, "minimise_adam")
    return minimise_f(
        f=f,
        vs=vs,
        iters=iters,
        rate=rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        local_rates=local_rates,
        trace=trace,
        names=names,
        jit=jit,
    )


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
        names = _convert_and_validate_names(names)

        # Extract variable container and auxilary arguments from argument specification.
        vs = convert(vs, tuple)
        vs, args = vs[0], vs[1:]

        # Run function once to ensure that all variables are initialised and
        # available.
        res = convert(f(vs, *args), tuple)
        val_init, args = res[0], res[1:]

        # SciPy doesn't perform zero iterations, so handle that edge case
        # manually.
        if iters == 0 or f_calls == 0:
            return B.squeeze((B.to_numpy(val_init),) + args)

        # Extract initial value.
        x0 = B.to_numpy(vs.get_latent_vector(*names))

        # The optimiser expects to get `float64`s.
        def _convert(*xs):
            return [B.cast(np.float64, B.to_numpy(x)) for x in xs]

        # Wrap the function and get the list of function evaluations.
        f_vals, f_wrapped = wrap_f(vs, names, f, jit, _convert)

        # Maintain a state for the auxilary arguments.
        state = {"args": args}

        # Wrap `f_wrapped` to take the auxilary from the global state and then update
        # the global state.

        def f_wrapped_self_passing(x):
            (obj_value, state["args"]), grad = f_wrapped(x, *state["args"])
            return obj_value, grad

        # Perform optimisation routine.
        def perform_minimisation(callback_=lambda _: None):
            return fmin_l_bfgs_b(
                func=f_wrapped_self_passing,
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

        vs.set_latent_vector(x_opt, *names)  # Assign optimum.
        args = state["args"]  # Get auxilary arguments at final state.

        return B.squeeze((val_opt,) + args)  # Return optimal value.

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
        names = _convert_and_validate_names(names)

        # Extract variable container and auxilary arguments from argument specification.
        vs = convert(vs, tuple)
        vs, args = vs[0], vs[1:]

        # Run function once to ensure that all variables are initialised and
        # available.
        res = convert(f(vs, *args), tuple)
        val_init, args = res[0], res[1:]

        # Handle the edge case of zero iterations.
        if iters == 0:
            return B.squeeze((B.to_numpy(val_init),) + args)

        # Extract initial value.
        x0 = B.to_numpy(vs.get_latent_vector(*names))

        # Wrap the function.
        _, f_wrapped = wrap_f(vs, names, f, jit, B.to_numpy)

        # Maintain a state for the auxilary arguments.
        state = {"args": args}

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
                (obj_value, state["args"]), grad = f_wrapped(x, *state["args"])
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

        vs.set_latent_vector(x_opt, *names)  # Assign optimum.
        args = state["args"]  # Get auxilary arguments at final state.

        return B.squeeze((obj_value,) + args)  # Return last objective value.

    return _minimise_adam


def exception(x, args, e):
    """In the case that an exception is raised during function evaluation,
    print a warning and return NaN for the function value and gradient.

    Args:
        x (tensor): Current input.
        args (tuple): Current auxilary arguments.
        e (:class:`Exception`): Caught exception.

    Returns:
        tuple: Return value containing NaN for the objective value and NaNs for the
            gradient.
    """
    with out.Section("Caught exception during function evaluation"):
        out.out(traceback.format_exc().strip())
    grad_nan = np.empty(x.shape)
    grad_nan[:] = np.nan
    return (np.nan, args), grad_nan
