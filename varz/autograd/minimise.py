import logging

import lab.autograd as B
from autograd import grad as only_grad, value_and_grad
from plum import convert

from ..minimise import make_l_bfgs_b, make_adam, exception

__all__ = ["minimise_l_bfgs_b", "minimise_adam"]

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f, jit, _convert):
    if jit:
        raise ValueError("There is no JIT for AutoGrad.")

    # Differentiable assignments will overwrite the variables, so make a copy.
    vs_copy = vs.copy()

    # Keep track of function evaluations.
    f_evals = []

    def f_vectorised(x, *args):
        vs_copy.set_latent_vector(x, *names, differentiable=True)
        out = convert(f(vs_copy, *args), tuple)
        return out[0], out[1:]

    def f_wrapped(x, *args):
        x = B.cast(vs.dtype, x)

        # Compute objective function value and gradient.
        try:
            if args == ():
                # Don't need to update `args`.
                obj_value, grad = value_and_grad(lambda x_: f_vectorised(x_)[0])(x)
            else:
                # AutoGrad doesn't allow us to only compute the gradient with respect
                # to the first output, so we have to do it this way, incurring an
                # extra forward passs...
                obj_value, new_args = f_vectorised(x, *args)
                grad = only_grad(lambda x_: f_vectorised(x_, *args)[0])(x)
                args = new_args

        except Exception as e:
            return exception(x, args, e)

        # Perform requested conversion.
        obj_value, grad = _convert(obj_value, grad)

        f_evals.append(obj_value)
        return (obj_value, args), grad

    return f_evals, f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
minimise_adam = make_adam(_wrap_f)
