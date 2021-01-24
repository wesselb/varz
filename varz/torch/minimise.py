import logging

import lab as B
import torch
import torch.autograd as autograd

from ..minimise import make_l_bfgs_b, make_adam, exception

__all__ = ["minimise_l_bfgs_b", "minimise_adam"]

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f, jit):
    # Differentiable assignments will overwrite the variables, so make a copy with
    # detached variables.
    vs_copy = vs.copy(detach=True)

    # Keep track of function evaluations.
    f_evals = []

    def f_vectorised(x):
        vs_copy.set_vector(x, *names, differentiable=True)
        if jit:
            with B.lazy_shapes:
                return f(vs_copy)
        else:
            return f(vs_copy)

    if jit:
        # It appears that PyTorch is not able to JIT through `autograd.grad`, so we
        # must already JIT here.
        f_vectorised = torch.jit.trace(f_vectorised, vs_copy.get_vector(*names))

    def f_value_and_grad(x):
        x.requires_grad_(True)
        obj_value = f_vectorised(x)
        grad = autograd.grad(obj_value, x)[0]
        return obj_value, grad

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
