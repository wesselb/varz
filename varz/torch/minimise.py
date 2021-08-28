import logging

import lab as B
from plum import convert

from ..minimise import make_l_bfgs_b, make_adam, exception

__all__ = ["minimise_l_bfgs_b", "minimise_adam"]

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f, jit, _convert):
    # Differentiable assignments will overwrite the variables, so make a copy with
    # detached variables.
    vs_copy = vs.copy(detach=True)

    # Keep track of function evaluations.
    f_evals = []

    def f_vectorised(x, *args):
        vs_copy.set_latent_vector(x, *names, differentiable=True)
        out = convert(f(vs_copy, *args), tuple)
        return out[0], out[1:]

    if jit:
        # It appears that PyTorch is not able to JIT through `autograd.grad`, so we
        # must already JIT here.
        f_vectorised = B.jit(f_vectorised)

    def f_value_and_grad(x, *args):
        x.requires_grad_(True)
        obj_value, args = f_vectorised(x, *args)
        obj_value.backward()
        return (obj_value, args), x.grad

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
