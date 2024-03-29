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
import wbml.out as out
from plum import isinstance
from varz import Vars
from varz.minimise import _convert_and_validate_names

from .util import OutStream, Value, approx

_rate = 5e-2
_minimise_method_params = []
for dtype_name in ["float32", "float64"]:
    _minimise_method_params.extend(
        [
            (getattr(np, dtype_name), varz.minimise_l_bfgs_b, {}),
            (
                getattr(np, dtype_name),
                varz.minimise_adam,
                {"rate": _rate, "local_rates": False},
            ),
            (
                getattr(np, dtype_name),
                varz.minimise_adam,
                {"rate": _rate, "local_rates": True},
            ),
        ]
    )
    for backend in [tf, torch, jnp]:
        _minimise_method_params.extend(
            [
                (
                    getattr(backend, dtype_name),
                    varz.minimise_l_bfgs_b,
                    {"jit": True},
                ),
                (
                    getattr(backend, dtype_name),
                    varz.minimise_adam,
                    {"rate": _rate, "jit": True, "local_rates": False},
                ),
                (
                    getattr(backend, dtype_name),
                    varz.minimise_adam,
                    {"rate": _rate, "jit": True, "local_rates": True},
                ),
            ]
        )


@pytest.fixture(params=_minimise_method_params)
def minimise_method(request):
    yield request.param


def test_convert_and_validate_names():
    assert _convert_and_validate_names(None) == []
    assert _convert_and_validate_names([]) == []
    assert _convert_and_validate_names("test") == ["test"]
    assert _convert_and_validate_names(["a", "b"]) == ["a", "b"]
    with pytest.raises(ValueError):
        _convert_and_validate_names(["a", ["b"]])
    with pytest.raises(ValueError):
        _convert_and_validate_names([1, "a"])


def test_docstring(minimise_method):
    assert minimise_method.__doc__ is not None


def test_no_jit_autograd():
    vs = Vars(dtype=np.float64)

    def f(vs_):
        return vs_.ubnd(name="x")

    with pytest.raises(ValueError):
        varz.autograd.minimise_l_bfgs_b(f, vs, jit=True)
    with pytest.raises(ValueError):
        varz.autograd.minimise_adam(f, vs, jit=True)


def test_minimise(minimise_method):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)

    # Initialise a variable that is not used.
    vs.ubnd(name="other")

    # Define some objective.
    def f(vs_):
        return (-3 - vs_.pos(name="x", init=1.0)) ** 2

    # Minimise it, until convergence.
    val_opt = minimise(f, vs, iters=2000, **kw_args)

    # Check for equality up to two digits.
    approx(val_opt, 9, atol=1e-2)
    approx(vs["x"], 0, atol=1e-2)


def test_minimise_auxilary_argument(minimise_method):
    # Perform the above test, but also pass an auxilary argument around.
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)

    # Again, initialise a variable that is not used.
    vs.ubnd(name="other")

    # Again, define some objective.
    def f(vs_, state):
        return (-3 - vs_.pos(name="x", init=1.0)) ** 2, state + 1

    # Minimise it, until convergence, but now also get the final state.
    val_opt, final_state = minimise(f, (vs, B.cast(dtype, 1)), iters=2000, **kw_args)

    # Check for equality up to two digits.
    approx(val_opt, 9, atol=1e-2)
    approx(vs["x"], 0, atol=1e-2)

    # Check that the internal state was passed around.
    assert final_state > 5


class _CaptureOut:
    def __init__(self):
        self._output = ""

    def __enter__(self):
        out.streams.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert out.streams[-1] is self
        del out.streams[-1]

    def write(self, x):
        self._output += x

    def __contains__(self, x):
        return x in self._output


@pytest.mark.parametrize("trace", [True, False])
def test_minimise_callback(minimise_method, trace):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)

    kw_args = dict(kw_args)  # Copy to prevent mutation.
    kw_args["trace"] = trace

    # Skip the methods which use the JIT.
    if "jit" in kw_args and kw_args["jit"]:
        return

    # Initialise a variable that is not used.
    vs.ubnd(name="other")

    # Tracking lists:
    objs = []
    objs_manual = []
    xs = []
    xs_manual = []

    # Define some objective.
    def f(vs_, prev_x):
        x = vs_.pos(name="x", init=1.0)
        obj = (-3 - x) ** 2
        # Track manually:
        objs_manual.append(obj)
        xs_manual.append(x)
        return obj, x

    # Define callback.
    def callback(obj, x):
        # Track in the way one should:
        objs.append(obj)
        xs.append(x)
        # Report additional information:
        return {"additional entry": xs[-1]}

    # Minimise it a bit.
    with OutStream() as stream:
        val_opt = minimise(f, (vs, 0), iters=20, callback=callback, **kw_args)

    # Every value except for the first one must be tracked: the first one is an initial
    # evaluation of the function. If we used AutoGrad, due to the extra forward pass,
    # every value is tracked twice... Make sure that everything is comparable by
    # converting to NumPy.
    objs_manual = [B.to_numpy(x) for x in objs_manual[1:]]
    xs_manual = [B.to_numpy(x) for x in xs_manual[1:]]
    if isinstance(dtype, B.NPDType):
        objs_manual = objs_manual[::2]
        xs_manual = xs_manual[::2]
    assert objs == objs_manual
    assert xs == xs_manual

    if trace:
        # Check that the additional information was also printed.
        assert "additional entry" in stream.output
    else:
        # No additinal information should have been printed.
        assert "additional entry" not in stream.output


def test_minimise_disconnected_gradient(minimise_method):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)
    vs.ubnd(name="x")

    # Check that optimiser runs for objective that returns the constant zero.
    minimise(lambda v: B.cast(v.dtype, 0), vs, **kw_args)


def test_minimise_exception(minimise_method):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)

    # Don't use the JIT in this test, because the tracing will fail due to the
    # randomness.
    kw_args = dict(kw_args)  # Copy to prevent mutation.
    kw_args["jit"] = False
    kw_args["iters"] = 100

    first_call = Value(True)

    # Define an objective that sometimes fails after the first call.
    def f(vs_):
        if first_call.val:
            first_call.val = False
        else:
            if np.random.rand() > 0.5:
                raise Exception("Fail!")
        return vs_.ubnd(name="x", init=5.0) ** 2

    # Check that the optimiser runs.
    minimise(f, vs, **kw_args)


def test_minimise_zero_calls(minimise_method):
    dtype, minimise, kw_args = minimise_method
    vs = Vars(dtype=dtype)

    calls = Value(0)

    def f(vs_):
        calls.val += 1
        return vs_.ubnd(name="x", init=5.0) ** 2

    # Check that running the optimiser for zero iterations only incurs a
    # single call.
    minimise(f, vs, iters=0, **kw_args)
    assert calls.val == 1


def test_minimise_trace(minimise_method):
    dtype, minimise, kw_args = minimise_method

    def f(vs_):
        return vs_.ubnd(name="x") ** 2

    # Test that `trace = False` prints nothing.
    with OutStream() as stream:
        minimise(f, Vars(dtype=dtype), trace=False, **kw_args)
        assert stream.output == ""

    # Test that `trace = False` prints something.
    with OutStream() as stream:
        minimise(f, Vars(dtype=dtype), trace=True, **kw_args)
        assert stream.output != ""
