import logging

import lab.tensorflow as B
import tensorflow as tf
from plum import convert

from ..minimise import make_l_bfgs_b, make_adam, exception

__all__ = ["minimise_l_bfgs_b", "minimise_adam"]

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f, jit, _convert):
    # Differentiable assignments will overwrite the variables, so make a copy.
    vs_copy = vs.copy()

    # Keep track of function evaluations.
    f_evals = []

    def f_vectorised(x, *args):
        vs_copy.set_latent_vector(x, *names, differentiable=True)
        out = convert(f(vs_copy, *args), tuple)
        return out[0], out[1:]

    def f_value_and_grad(x, *args):
        with tf.GradientTape() as t:
            t.watch(x)
            obj_value, args = f_vectorised(x, *args)
            grad = t.gradient(obj_value, x, unconnected_gradients="zero")
        return (obj_value, args), grad

    if jit:
        f_value_and_grad = B.jit(f_value_and_grad)

    def f_wrapped(x, *args):
        x = B.cast(vs.dtype, x)

        # Compute objective function value and gradient.
        try:
            (obj_value, args), grad = f_value_and_grad(x, *args)
        except Exception as e:
            return exception(x, args, e)

        # Perform requested conversion.
        obj_value, grad = _convert(obj_value, grad)

        f_evals.append(obj_value)
        return (obj_value, args), grad

    return f_evals, f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
minimise_adam = make_adam(_wrap_f)
