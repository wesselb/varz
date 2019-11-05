import logging

from autograd import value_and_grad

from ..minimise import make_l_bfgs_b, make_adam, exception

__all__ = ['minimise_l_bfgs_b', 'minimise_adam']

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f):
    # Differentiable assignments will overwrite the variables, so make a copy.
    vs_copy = vs.copy()

    # Keep track of function evaluations.
    f_evals = []

    def f_vectorised(x):
        vs_copy.set_vector(x, *names, differentiable=True)
        return f(vs_copy)

    def f_wrapped(x):
        # Compute objective function value.
        try:
            obj_value, grad = value_and_grad(f_vectorised)(x)
        except Exception as e:
            return exception(x, e)

        f_evals.append(obj_value)
        return obj_value, grad

    return f_evals, f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
minimise_adam = make_adam(_wrap_f)
