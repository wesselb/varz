import logging

import jax
import lab.jax as B
import numpy as np
from jax import value_and_grad

from ..minimise import make_l_bfgs_b, make_adam, exception

__all__ = ["minimise_l_bfgs_b", "minimise_adam"]

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f, jit):
    # Differentiable assignments will overwrite the variables, so make a copy.
    vs_copy = vs.copy()

    # Keep track of function evaluations.
    f_evals = []

    def f_vectorised(x):
        vs_copy.set_vector(x, *names, differentiable=True)
        return f(vs_copy)

    if jit:
        f_value_and_grad = jax.jit(value_and_grad(f_vectorised))
    else:
        f_value_and_grad = value_and_grad(f_vectorised)

    def f_wrapped(x):
        x = B.cast(vs.dtype, x)

        # Compute objective function value and gradient.
        try:
            obj_value, grad = f_value_and_grad(x)
        except Exception as e:
            return exception(x, e)

        # Convert to NumPy.
        obj_value, grad = B.to_numpy(obj_value, grad)

        # The gradient may not have the right memory layout, which sometimes cannot
        # be adjusted. We therefore make a copy, which can always be freely manipulated.
        grad = np.array(grad)

        f_evals.append(obj_value)
        return obj_value, grad

    return f_evals, f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
minimise_adam = make_adam(_wrap_f)
