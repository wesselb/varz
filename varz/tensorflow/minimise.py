import logging

import lab.tensorflow as B
import tensorflow as tf

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

    def f_value_and_grad(x):
        with tf.GradientTape() as t:
            t.watch(x)
            obj_value = f_vectorised(x)
            grad = t.gradient(obj_value, x, unconnected_gradients="zero")
        return obj_value, grad

    if jit:
        f_value_and_grad = tf.function(f_value_and_grad, autograph=False)

    def f_wrapped(x):
        x = B.cast(vs.dtype, x)

        # Compute objective function value and gradient.
        try:
            obj_value, grad = f_value_and_grad(x)
        except Exception as e:
            return exception(x, e)

        # Convert to NumPy.
        obj_value, grad = B.to_numpy(obj_value, grad)

        f_evals.append(obj_value)
        return obj_value, grad

    return f_evals, f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
minimise_adam = make_adam(_wrap_f)
