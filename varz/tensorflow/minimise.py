import logging

import tensorflow as tf

from ..minimise import make_l_bfgs_b, make_adam, exception

__all__ = ['minimise_l_bfgs_b', 'minimise_adam']

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f):
    # Keep track of function evaluations.
    f_evals = []

    def f_wrapped(x):
        # Update to current point.
        vs.set_vector(x, *names)

        # Compute objective function value and gradient.
        try:
            with tf.GradientTape() as t:
                t.watch(vs.get_vars(*names))
                obj_value = f(vs)
                grads = t.gradient(obj_value, vs.get_vars(*names),
                                   unconnected_gradients='zero')
        except Exception as e:
            return exception(x, e)

        # Construct gradient.
        grad = vs.vector_packer.pack(*grads)

        f_evals.append(obj_value)
        return obj_value.numpy(), grad.numpy()

    return f_evals, f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
minimise_adam = make_adam(_wrap_f)
