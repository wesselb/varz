import numpy as np
import pytest
import torch
from varz import Vars
import lab as B

# noinspection PyUnresolvedReferences
from .util import (
    approx,
    approx,
    KV,
    dtype,
    vs,
    vs_source,
    assert_lower_triangular,
    assert_positive_definite,
    assert_orthogonal,
)


def test_get_latent_vars():
    vs = Vars(np.int)

    # This test also tests that `Vars.get_vars` always returns the collection
    # of variables in the right order. This is important for optimisation.

    # Initialise some variables.
    vs.ubnd(1, name="a")
    vs.ubnd(2, name="1/b")
    vs.ubnd(3, name="2/c")
    vs.unbounded(4, name="2/d")  # Test alias.

    # Test getting all.
    assert vs.get_latent_vars() == [1, 2, 3, 4]
    assert vs.get_latent_vars(return_indices=True) == [0, 1, 2, 3]

    # Test that names must exist.
    with pytest.raises(ValueError):
        vs.get_latent_vars("e")

    # Test some queries.
    assert vs.get_latent_vars("a") == [1]
    assert vs.get_latent_vars("a", "*/b") == [1, 2]
    assert vs.get_latent_vars("*/c", "a") == [1, 3]
    assert vs.get_latent_vars("*/c", "*/b", "a") == [1, 2, 3]

    assert vs.get_latent_vars("a", return_indices=True) == [0]
    assert vs.get_latent_vars("a", "*/b", return_indices=True) == [0, 1]
    assert vs.get_latent_vars("*/c", "a", return_indices=True) == [0, 2]
    assert vs.get_latent_vars("*/c", "*/b", "a", return_indices=True) == [0, 1, 2]

    # Test some more queries.
    assert vs.get_latent_vars("1/*") == [2]
    assert vs.get_latent_vars("2/*") == [3, 4]
    assert vs.get_latent_vars("1/*", "2/*") == [2, 3, 4]

    assert vs.get_latent_vars("1/*", return_indices=True) == [1]
    assert vs.get_latent_vars("2/*", return_indices=True) == [2, 3]
    assert vs.get_latent_vars("1/*", "2/*", return_indices=True) == [1, 2, 3]

    # Test even more queries.
    assert vs.get_latent_vars("*/b", "1/*") == [2]
    assert vs.get_latent_vars("a", "2/*") == [1, 3, 4]
    assert vs.get_latent_vars("a", "2/d", "2/*") == [1, 3, 4]
    assert vs.get_latent_vars("2/d", "2/c", "a", "1/*") == [1, 2, 3, 4]
    assert vs.get_latent_vars("1/*") == [2]
    assert vs.get_latent_vars("2/*") == [3, 4]
    assert vs.get_latent_vars("1/*", "2/*") == [2, 3, 4]

    assert vs.get_latent_vars("*/b", "1/*", return_indices=True) == [1]
    assert vs.get_latent_vars("a", "2/*", return_indices=True) == [0, 2, 3]
    assert vs.get_latent_vars("a", "2/d", "2/*", return_indices=True) == [0, 2, 3]
    inds = vs.get_latent_vars("2/d", "2/c", "a", "1/*", return_indices=True)
    assert inds == [0, 1, 2, 3]
    assert vs.get_latent_vars("1/*", return_indices=True) == [1]
    assert vs.get_latent_vars("2/*", return_indices=True) == [2, 3]
    assert vs.get_latent_vars("1/*", "2/*", return_indices=True) == [1, 2, 3]


def test_get_latent_vars_cache_clearing(vs):
    vs.ubnd(name="var_a")
    assert vs.get_latent_vars("var_*", return_indices=True) == [0]
    vs.ubnd(name="var_b")
    assert vs.get_latent_vars("var_*", return_indices=True) == [0, 1]
    vs.delete("var_b")
    assert vs.get_latent_vars("var_*", return_indices=True) == [0]


def test_get_latent_vars_visible(vs):
    vs.ubnd(1, name="x1")
    vs.ubnd(2, name="x2", visible=False)
    vs.ubnd(3, name="x3")

    assert vs.get_latent_vars() == [1, 3]
    assert vs.get_latent_vars(return_indices=True) == [0, 2]
    assert vs.get_latent_vars("x*") == [1, 3]
    assert vs.get_latent_vars("x*", return_indices=True) == [0, 2]


def test_get_latent_vars_exclusion(vs):
    vs.ubnd(1, name="x1")
    vs.ubnd(2, name="x2")
    vs.ubnd(3, name="y1")
    vs.ubnd(4, name="y2")

    assert vs.get_latent_vars() == [1, 2, 3, 4]
    assert vs.get_latent_vars("-y") == [1, 2, 3, 4]
    assert vs.get_latent_vars("-y1") == [1, 2, 4]
    assert vs.get_latent_vars("-y2") == [1, 2, 3]
    assert vs.get_latent_vars("-y*") == [1, 2]
    assert vs.get_latent_vars("*2") == [2, 4]
    assert vs.get_latent_vars("*2", "-y") == [2, 4]
    assert vs.get_latent_vars("*2", "-y1") == [2, 4]
    assert vs.get_latent_vars("*2", "-y2") == [2]
    assert vs.get_latent_vars("*2", "-y*") == [2]


def test_contains(vs):
    vs.ubnd(1, name="a")
    vs.ubnd(2, name="b")

    assert "a" in vs
    assert "b" in vs
    assert "c" not in vs


def test_delete(vs):
    vs.ubnd(1, name="a")
    vs.ubnd(2, name="b")
    vs.ubnd(3, name="c")

    assert "b" in vs.name_to_index
    assert len(vs.vars) == 3
    assert len(vs.transforms) == 3
    assert len(vs.inverse_transforms) == 3
    # Cache clearing is tested in :func:`test_get_latent_vars_cache_clearing`.

    vs.delete("b")

    assert vs["a"] == 1
    assert vs["c"] == 3

    assert "b" not in vs.name_to_index
    assert len(vs.vars) == 2
    assert len(vs.transforms) == 2
    assert len(vs.inverse_transforms) == 2

    vs.delete("a")

    assert vs["c"] == 3

    assert "a" not in vs.name_to_index
    assert len(vs.vars) == 1
    assert len(vs.transforms) == 1
    assert len(vs.inverse_transforms) == 1

    vs.ubnd(1, name="a")

    assert vs["a"] == 1
    assert vs["c"] == 3

    assert "a" in vs.name_to_index
    assert len(vs.vars) == 2
    assert len(vs.transforms) == 2
    assert len(vs.inverse_transforms) == 2


def test_unbounded(vs_source):
    for _ in range(10):
        vs_source.ubnd()
        vs_source.unbounded()


def test_unbounded_init(vs):
    vs.ubnd(1, name="x")
    approx(vs["x"], 1)

    # Test that explicit data type can be given. This should work the same
    # for all variable getters, so we only test it once.
    assert B.dtype(vs.ubnd(1, name="y", dtype=int)) == np.int64


def test_unbounded_assignment(vs):
    vs.ubnd(1, name="x")
    vs.assign("x", 2)
    approx(vs["x"], 2)


def check_visible(vs, method):
    method(name="w", visible=True)
    method(name="x", visible=False)
    method(name="y", visible=True)
    method(name="z", visible=False)
    assert vs.get_latent_vars(return_indices=True) == [0, 2]


def test_unbounded_visible(vs):
    check_visible(vs, vs.ubnd)


def test_positive(vs_source):
    for _ in range(10):
        assert vs_source.pos() >= 0
        assert vs_source.positive() >= 0


def test_positive_init(vs):
    vs.pos(1, name="x")
    approx(vs["x"], 1)


def test_positive_assignment(vs):
    vs.pos(1, name="x")
    vs.assign("x", 2)
    approx(vs["x"], 2)


def test_positive_visible(vs):
    check_visible(vs, vs.pos)


def test_bounded(vs_source):
    for _ in range(10):
        assert 10 <= vs_source.bnd(lower=10, upper=11) <= 11
        assert 10 <= vs_source.bounded(lower=10, upper=11) <= 11


def test_bounded_init(vs):
    vs.bnd(2, name="x", lower=1, upper=4)
    approx(vs["x"], 2)


def test_bounded_assignment(vs):
    vs.bnd(2, name="x", lower=1, upper=4)
    vs.assign("x", 3)
    approx(vs["x"], 3)


def test_bounded_monotonic(vs):
    vs.bnd(1, lower=0, upper=10)
    vs.bnd(2, lower=0, upper=10)
    vs.bnd(3, lower=0, upper=10)
    assert vs.vars[0] < vs.vars[1] < vs.vars[2]


def test_bounded_visible(vs):
    check_visible(vs, vs.bnd)


def test_lower_triangular(vs_source):
    for _ in range(10):
        assert B.shape(vs_source.tril(shape=(5, 5))) == (5, 5)
        assert_lower_triangular(vs_source.tril(shape=(5, 5)))
        assert_lower_triangular(vs_source.lower_triangular(shape=(5, 5)))


def test_lower_triangular_init(vs):
    x = vs.tril(shape=(5, 5))

    vs.tril(x, name="x")
    approx(vs["x"], x)


@pytest.mark.parametrize("shape", [None, (5,), (5, 6)])
def test_lower_triangular_shape(vs, shape):
    with pytest.raises(ValueError):
        vs.tril(shape=shape)


def test_lower_triangular_assignment(vs):
    x = vs.tril(shape=(5, 5))

    vs.tril(shape=(5, 5), name="x")
    vs.assign("x", x)
    approx(vs["x"], x)


def test_lower_triangular_visible(vs):
    check_visible(vs, lambda **kw_args: vs.tril(shape=(5, 5), **kw_args))


def test_positive_definite(vs_source):
    for _ in range(10):
        assert B.shape(vs_source.pd(shape=(5, 5))) == (5, 5)
        assert_positive_definite(vs_source.pd(shape=(5, 5)))
        assert_positive_definite(vs_source.positive_definite(shape=(5, 5)))


def test_positive_definite_init(vs):
    x = vs.pd(shape=(5, 5))

    vs.pd(x, name="x")
    approx(vs["x"], x)


@pytest.mark.parametrize("shape", [None, (5,), (5, 6)])
def test_positive_definite_shape(vs, shape):
    with pytest.raises(ValueError):
        vs.pd(shape=shape)


def test_positive_definite_assignment(vs):
    x = vs.pd(shape=(5, 5))

    vs.pd(shape=(5, 5), name="x")
    vs.assign("x", x)
    approx(vs["x"], x)


def test_positive_definite_visible(vs):
    check_visible(vs, lambda **kw_args: vs.pd(shape=(5, 5), **kw_args))


@pytest.mark.parametrize("method", ["svd", "expm", "cayley"])
def test_orthogonal(vs_source, method):
    for i in range(10):
        assert B.shape(vs_source.orth(shape=(5, 5), method=method)) == (5, 5)
        assert_orthogonal(vs_source.orth(shape=(5, 5), method=method))
        assert_orthogonal(vs_source.orthogonal(shape=(5, 5), method=method))


def test_orthogonal_method(vs_source):
    with pytest.raises(ValueError):
        vs_source.orth(shape=(5, 5), method="bla")


@pytest.mark.parametrize("method", ["expm", "cayley"])
def test_orthogonal_init(vs, method):
    x = vs.orth(shape=(5, 5), method=method)

    vs.orth(x, name="x", method=method)
    approx(vs["x"], x)


@pytest.mark.parametrize("shape", [(5, 5), (3, 7), (7, 3)])
def test_orthogonal_init_svd(vs, shape):
    x = vs.orth(shape=shape, method="svd")

    vs.orth(x, name="x", method="svd")
    approx(vs["x"], x)


@pytest.mark.parametrize("method", ["expm", "cayley"])
@pytest.mark.parametrize("shape", [None, (5,), (5, 6)])
def test_orthogonal_shape(vs, shape, method):
    with pytest.raises(ValueError):
        vs.orth(shape=shape, method=method)


@pytest.mark.parametrize("shape", [None, (5,)])
def test_orthogonal_shape_svd(vs, shape):
    with pytest.raises(ValueError):
        vs.orth(shape=shape, method="svd")


@pytest.mark.parametrize("method", ["svd", "expm", "cayley"])
def test_orthogonal_assignment(vs, method):
    x = vs.orth(shape=(5, 5), method=method)

    vs.orth(shape=(5, 5), name="x", method=method)
    vs.assign("x", x)
    approx(vs["x"], x)


def test_orthogonal_visible(vs):
    check_visible(vs, lambda **kw_args: vs.orth(shape=(5, 5), **kw_args))


def test_init_shape_ambiguity(vs):
    with pytest.raises(ValueError):
        vs.ubnd(np.ones(1), shape=())


def test_init_broadcasting(vs):
    vs.ubnd(2, shape=(2,), name="x")
    approx(vs["x"], 2 * B.ones(2))


def test_init_shape_checking(vs):
    vs.ubnd(np.array([1]), shape=(1,), name="x")
    vs.ubnd(np.array([1]), shape=[1], name="y")


def test_get_set_latent_vector(dtype):
    vs = Vars(dtype=dtype)

    # Test stacking a matrix and a vector.
    vs.ubnd(shape=(2,), name="a", init=np.array([1, 2]))
    vs.ubnd(shape=(2, 2), name="b", init=np.array([[3, 4], [5, 6]]))
    approx(vs.get_latent_vector("a", "b"), [1, 2, 3, 4, 5, 6])

    # Test setting elements.
    vs.set_latent_vector(np.array([6, 5, 4, 3, 2, 1]), "a", "b")
    approx(vs["a"], np.array([6, 5]))
    approx(vs["b"], np.array([[4, 3], [2, 1]]))

    # Test setting elements in a differentiable way. This should allow for
    # any values.
    vs.set_latent_vector(
        np.array(["1", "2", "3", "4", "5", "6"]), "a", "b", differentiable=True
    )
    assert np.all(vs["a"] == ["1", "2"])
    assert np.all(vs["b"] == np.array([["3", "4"], ["5", "6"]]))


def test_copy_torch():
    vs = Vars(torch.float64)

    # Create a variable.
    vs.pos(1, name="a")

    # Make a normal and detached copy.
    vs_copy = vs.copy()
    vs_detached = vs.copy(detach=True)

    # Require gradients for both.
    vs.requires_grad(True)
    vs_detached.requires_grad(True)

    # Do a backward pass.
    (vs_detached["a"] ** 2).backward()

    # Check that values are equal, but gradients only computed for one.
    assert vs["a"] == 1
    assert vs.get_latent_vars("a")[0].grad is None
    assert vs_detached["a"] == 1
    assert vs_detached.get_latent_vars("a")[0].grad == 2

    # Check that copied fields are, in fact, copies.
    # is also copied.
    del vs.transforms[:]
    del vs.inverse_transforms[:]
    del vs.vars[:]
    vs.name_to_index.clear()

    for vs_copied in [vs_copy, vs_detached]:
        assert len(vs_copied.transforms) > 0
        assert len(vs_copied.inverse_transforms) > 0
        assert len(vs_copied.vars) > 0
        assert len(vs_copied.name_to_index) > 0


def test_copy_f():
    vs = Vars(np.float64)
    vs.unbounded(1, name="a")
    vs.unbounded(2, name="b")

    # Make a copy and apply a function that sets everything to zero.
    vs_copy = vs.copy(f=lambda x: 0 * x)

    assert vs["a"] == 1
    assert vs["b"] == 2
    assert vs_copy["a"] == 0
    assert vs_copy["b"] == 0


def test_requires_grad_detach_vars_torch():
    vs = Vars(torch.float64)
    vs.pos(1, name="a")

    # Test that gradients need to first be required.
    with pytest.raises(RuntimeError):
        (2 * vs["a"]).backward()

    # Test that gradients can be required and are then computed.
    vs.requires_grad(True)
    (2 * vs["a"]).backward()
    assert vs.vars[0].grad is not None

    # Test that variables can be detached.
    vs.pos(1, name="b")
    result = 2 * vs["b"]
    vs.detach()
    with pytest.raises(RuntimeError):
        result.backward()


def test_source():
    vs = Vars(np.float32, source=np.ones(10))

    assert vs.ubnd() == 1.0
    approx(vs.pos(shape=(5,)), np.exp(np.ones(5, dtype=np.float32)))
    approx(vs.pos(), np.exp(np.float32(1.0)))
    with pytest.raises(ValueError):
        vs.pos(shape=(5,))

    # Test that the source variables are casted to the right data type.

    vs = Vars(np.float32, source=np.array([1]))
    assert vs.ubnd().dtype == np.float32

    vs = Vars(np.float64, source=np.array([1]))
    assert vs.ubnd().dtype == np.float64


def test_names(vs):
    assert vs.names == []
    vs.ubnd()
    assert vs.names == []
    vs.pos(name="a")
    assert vs.names == ["a"]
    vs.bnd(name="b")
    assert vs.names == ["a", "b"]
    vs.ubnd(name="c")
    assert vs.names == ["a", "b", "c"]


def test_print(vs):
    with KV() as mock:
        vs.print()
        assert mock.keys == []
        assert mock.values == []

        vs.ubnd()
        vs.print()
        assert mock.keys == []
        assert mock.values == []

        vs.print()
        vs.ubnd(1, name="a")
        vs.print()
        assert mock.keys == ["a"]
        assert mock.values == [1]

        vs.ubnd(2, name="b")
        vs.print()
        assert mock.keys == ["a", "a", "b"]
        assert mock.values == [1, 1, 2]

        vs.ubnd(3, name="c")
        vs.print()
        assert mock.keys == ["a", "a", "b", "a", "b", "c"]
        assert mock.values == [1, 1, 2, 1, 2, 3]
