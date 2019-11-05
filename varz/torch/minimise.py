import logging

import torch

from ..minimise import make_l_bfgs_b, make_adam, exception

__all__ = ['minimise_l_bfgs_b', 'minimise_adam']

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f):
    # Differentiable assignments will overwrite the variables, so make a copy
    # with detached variables.
    vs_copy = vs.copy(detach=True)

    # Keep track of function evaluations.
    f_evals = []

    def f_wrapped(x):
        x_torch = torch.tensor(x, requires_grad=True)

        # Compute objective function value.
        try:
            vs_copy.set_vector(x_torch, *names, differentiable=True)
            obj_value = f(vs_copy)
            obj_value.backward()
            obj_value = obj_value.detach_().numpy()
        except Exception as e:
            return exception(x, e)

        # Extract gradient.
        grad = x_torch.grad.detach_().numpy()

        f_evals.append(obj_value)
        return obj_value, grad

    return f_evals, f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
minimise_adam = make_adam(_wrap_f)
