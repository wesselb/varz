# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import tensorflow as tf

from ..minimise import make_l_bfgs_b, exception

__all__ = ['minimise_l_bfgs_b']

log = logging.getLogger(__name__)


def _wrap_f(vs, names, f):
    # Differentiable assignments will overwrite the variables, so make a copy.
    vs_copy = vs.copy()

    def f_wrapped(x):
        x_tf = tf.constant(x)

        # Compute objective function value and gradient.
        try:
            with tf.GradientTape() as t:
                t.watch(x_tf)
                vs_copy.set_vector(x_tf, *names, differentiable=True)
                obj_value = f(vs_copy)
                grad = t.gradient(obj_value, x_tf, unconnected_gradients='zero')
        except RuntimeError as e:
            return exception(x, e)

        return obj_value.numpy(), grad.numpy()

    return f_wrapped


minimise_l_bfgs_b = make_l_bfgs_b(_wrap_f)
