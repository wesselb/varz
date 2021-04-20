import jax.numpy as jnp
import lab as B
import numpy as np
import pytest
import tensorflow as tf
import torch

import varz.autograd
import varz.jax
import varz.tensorflow
import varz.torch
from varz import Vars
from .util import approx, Value, OutStream


@pytest.fixture(
    params=[
        (np.float64, varz.autograd.minimise_l_bfgs_b, {}),
        (np.float64, varz.autograd.minimise_adam, {"rate": 1e-1, "local_rates": False}),
        (np.float64, varz.autograd.minimise_adam, {"rate": 1e-1, "local_rates": True}),
    ]
    + sum(
        [
            [
                (tf.float64, varz.tensorflow.minimise_l_bfgs_b, {"jit": jit}),
                (
                    tf.float64,
                    varz.tensorflow.minimise_adam,
                    {"rate": 1e-1, "jit": jit, "local_rates": False},
                ),
                (
                    tf.float64,
                    varz.tensorflow.minimise_adam,
                    {"rate": 1e-1, "jit": jit, "local_rates": True},
                ),
                (torch.float64, varz.torch.minimise_l_bfgs_b, {"jit": jit}),
                (
                    torch.float64,
                    varz.torch.minimise_adam,
                    {"rate": 1e-1, "jit": jit, "local_rates": False},
                ),
                (
                    torch.float64,
                    varz.torch.minimise_adam,
                    {"rate": 1e-1, "jit": jit, "local_rates": True},
                ),
                (jnp.float64, varz.jax.minimise_l_bfgs_b, {"jit": jit}),
                (
                    jnp.float64,
                    varz.jax.minimise_adam,
                    {"rate": 1e-1, "jit": jit, "local_rates": False},
                ),
                (
                    jnp.float64,
                    varz.jax.minimise_adam,
                    {"rate": 1e-1, "jit": jit, "local_rates": True},
                ),
            ]
            for jit in [False, True]
        ],
        [],
    )
)
def minimise_method(request):
    yield request.param


def test_docstring(minimise_method):
    assert minimise_method.__doc__ is not None


def test_no_jit_autograd():
    vs = Vars(dtype=np.float64)

    def f(vs_):
        return vs_.get(name="x")

    with pytest.raises(ValueError):
        varz.autograd.minimise_l_bfgs_b(f, vs, jit=True)
    with pytest.raises(ValueError):
        varz.autograd.minimise_adam(f, vs, jit=True)


def test_minimise(minimise_method):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)

    # Initialise a variable that is not used.
    vs.get(name="other")

    # Define some objective.
    def f(vs_):
        return (-3 - vs_.pos(name="x", init=5.0)) ** 2

    # Minimise it, until convergence.
    val_opt = minimise(f, vs, iters=10000, **kw_args)

    # Check for equality up to three digits.
    approx(val_opt, 9, atol=1e-3)
    approx(vs["x"], 0, atol=1e-3)


def test_minimise_disconnected_gradient(minimise_method):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)
    vs.get(name="x")

    # Check that optimiser runs for objective that returns the constant zero.
    minimise(lambda v: B.cast(v.dtype, 0), vs, **kw_args)


def test_minimise_exception(minimise_method):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)

    # Skip this tests when the JIT is used, because tracing may then fail.
    if "jit" in kw_args and kw_args["jit"]:
        return

    first_call = Value(True)

    # Define an objective that sometimes fails after the first call.
    def f(vs_):
        if first_call.val:
            first_call.val = False
        else:
            if np.random.rand() > 0.5:
                raise Exception("Fail!")
        return vs_.get(name="x", init=5.0) ** 2

    # Check that the optimiser runs.
    minimise(f, vs, **kw_args)


def test_minimise_zero_calls(minimise_method):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)

    calls = Value(0)

    def f(vs_):
        calls.val += 1
        return vs_.get(name="x", init=5.0) ** 2

    # Check that running the optimiser for zero iterations only incurs a
    # single call.
    minimise(f, vs, iters=0, **kw_args)
    assert calls.val == 1


def test_minimise_trace(minimise_method):
    dtype, minimise, kw_args = minimise_method

    def f(vs_):
        return vs_.get(name="x") ** 2

    # Test that `trace=False` prints nothing.
    with OutStream() as stream:
        minimise(f, Vars(dtype=dtype), trace=False, **kw_args)
        assert stream.output == ""

    # Test that `trace=False` prints something.
    with OutStream() as stream:
        minimise(f, Vars(dtype=dtype), trace=True, **kw_args)
        assert stream.output != ""
