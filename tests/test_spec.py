import pytest

from varz import (
    sequential,
    namespace,
    Unbounded,
    Positive,
    Bounded,
    LowerTriangular,
    PositiveDefinite,
    Orthogonal,
    parametrised,
)
from varz.spec import _extract_prefix_and_f

# noinspection PyUnresolvedReferences
from .util import (
    vs,
    assert_lower_triangular,
    assert_positive_definite,
    assert_orthogonal,
)


def test_extract_prefix_and_f():
    def f():
        pass

    assert _extract_prefix_and_f(f, default="var") == ("var", f)
    assert _extract_prefix_and_f("", default="var") == ("", None)
    assert _extract_prefix_and_f("test", default="var") == ("test", None)


@pytest.mark.parametrize(
    "decorator, names",
    [
        (sequential, ["z", "var0", "var1", "var2"]),
        (sequential(), ["z", "var0", "var1", "var2"]),
        (sequential("x"), ["z", "x0", "x1", "x2"]),
    ],
)
def test_sequential(vs, decorator, names):
    vs.ubnd(0, name="z")

    @decorator
    def f(x, vs_, y):
        assert x == 1
        assert y == 2
        return vs_["z"], vs_.ubnd(), vs_.pos(), vs_.bnd(lower=10, upper=11)

    # Test that the same variables are retrieved.
    assert f(1, vs, 2) == f(1, vs, 2)
    assert vs.names == names
    assert f(1, vs, 2) == f(1, vs, 2)
    assert vs.names == names

    # Test correctness of variables.
    assert f(1, vs, 2)[0] == 0
    assert f(1, vs, 2)[2] >= 0
    assert 10 <= f(1, vs, 2)[3] <= 11


def test_sequential_unbounded(vs):
    @sequential
    def f(vs_):
        return vs_.ubnd(1)

    assert f(vs) == 1


def test_sequential_positive(vs):
    @sequential
    def f(vs_):
        return vs_.pos()

    assert f(vs) > 0


def test_sequential_bounded(vs):
    @sequential
    def f(vs_):
        return vs_.bnd(lower=10, upper=11)

    assert 10 <= f(vs) <= 11


def test_sequential_lower_triangular(vs):
    @sequential
    def f(vs_):
        return vs_.tril(shape=(5, 5))

    assert_lower_triangular(f(vs))


def test_sequential_positive_definite(vs):
    @sequential
    def f(vs_):
        return vs_.pd(shape=(5, 5))

    assert_positive_definite(f(vs))


def test_sequential_orthogonal(vs):
    @sequential
    def f(vs_):
        return vs_.orth(shape=(5, 5))

    assert_orthogonal(f(vs))


def test_namespace(vs):
    @namespace("test")
    def f(vs_):
        return [
            vs_.ubnd(name="a"),
            vs_.ubnd(),
            vs_.ubnd(name="b"),
        ]

    f(vs)
    names = ["test.a", "test.b"]

    # Check that the same variables are retrieved.
    assert f(vs)[0] == f(vs)[0] and f(vs)[2] == f(vs)[2]
    assert vs.names == names
    assert f(vs)[0] == f(vs)[0] and f(vs)[2] == f(vs)[2]
    assert vs.names == names


def test_namespace_sequential(vs):
    @namespace("test")
    @sequential
    def f(vs_):
        vs_.ubnd(name="a")
        vs_.ubnd()
        vs_.ubnd(name="b")

    f(vs)
    names = ["test.a", "test.var0", "test.b"]

    # Check that the same variables are retrieved.
    assert f(vs) == f(vs)
    assert vs.names == names
    assert f(vs) == f(vs)
    assert vs.names == names


def test_struct(vs):
    def f(vs_):
        params = vs_.struct
        res = []

        res.append(params.x1.positive())
        res.append(params.x1.positive(name="var1"))
        res.append(params.y1.positive())
        res.append(params.y1.positive(name="var2"))
        res.append(params["key1"].positive())
        res.append(params["key1"].positive(name="var3"))

        for pi, i in zip(params, [4, 5, 6]):
            res.append(pi.bounded())
            res.append(pi.bounded(name=f"var{i}"))

        params = params.level

        res.append(params.x2.positive())
        res.append(params.x2.positive(name="var7"))
        res.append(params.y2.positive())
        res.append(params.y2.positive(name="var8"))
        res.append(params["key2"].positive())
        res.append(params["key2"].positive(name="var9"))

        for pi, i in zip(params, [10, 11, 12]):
            res.append(pi.bounded())
            res.append(pi.bounded(name=f"var{i}"))

        return res

    f(vs)
    names = [
        "x1",
        "x1.var1",
        "y1",
        "y1.var2",
        "[key1]",
        "[key1].var3",
        "[0]",
        "[0].var4",
        "[1]",
        "[1].var5",
        "[2]",
        "[2].var6",
        "level.x2",
        "level.x2.var7",
        "level.y2",
        "level.y2.var8",
        "level[key2]",
        "level[key2].var9",
        "level[0]",
        "level[0].var10",
        "level[1]",
        "level[1].var11",
        "level[2]",
        "level[2].var12",
    ]

    # Check that the same variables are retrieved.
    assert f(vs) == f(vs)
    assert vs.names == names
    assert f(vs) == f(vs)
    assert vs.names == names

    # Check extraction by calling.
    assert vs.struct.x1.positive() == vs.struct.x1()


@pytest.mark.parametrize(
    "decorator, types, names",
    [
        (
            parametrised,
            [Unbounded, Positive, Bounded(lower=10, upper=11)],
            ["w", "x", "y", "z"],
        ),
        (
            parametrised(),
            [Unbounded, Positive, Bounded(lower=10, upper=11)],
            ["w", "x", "y", "z"],
        ),
        (
            parametrised("var_"),
            [Unbounded, Positive, Bounded(lower=10, upper=11)],
            ["w", "var_x", "var_y", "var_z"],
        ),
    ],
)
def test_parametrised(vs, decorator, types, names):
    vs.ubnd(0, name="w")

    @decorator
    def f(a, x: types[0], vs_, y: types[1], b, z: types[2], c=None):
        assert a == 1
        assert b == 2
        assert c is None
        return vs_["w"], x, y, z

    # Test that the same variables are retrieved
    assert f(1, vs, 2) == f(1, vs, 2)
    assert vs.names == names
    assert f(1, vs, 2) == f(1, vs, 2)
    assert vs.names == names

    # Test correctness of variables.
    assert f(1, vs, 2)[0] == 0
    assert f(1, vs, 2)[2] >= 0
    assert 10 <= f(1, vs, 2)[3] <= 11


def test_parametrised_arguments(vs):
    @parametrised
    def f(vs_, x: Unbounded, a, y: Unbounded, b=2, z: Unbounded = 10, c=3):
        return a, b, c

    with pytest.raises(ValueError):
        f(vs)

    assert f(vs, 1) == (1, 2, 3)
    assert f(vs, a=1) == (1, 2, 3)

    assert f(vs, 1, 4) == (1, 4, 3)
    assert f(vs, 1, b=4) == (1, 4, 3)
    assert f(vs, a=1, b=4) == (1, 4, 3)

    assert f(vs, 1, 4, 5) == (1, 4, 5)
    assert f(vs, 1, 4, c=5) == (1, 4, 5)
    assert f(vs, a=1, b=4, c=5) == (1, 4, 5)

    # Test unspecified positional argument.
    with pytest.raises(ValueError):
        f(vs, 1, 2, 3, 4)

    # Test unspecified keyword argument.
    with pytest.raises(ValueError):
        f(vs, 1, 2, d=3)


def test_parametrised_variable_container(vs):
    @parametrised
    def f(x, y, z: Unbounded):
        pass

    # Test that a variable container must be present.
    with pytest.raises(ValueError):
        f(1, 2)

    # Test that only one variable container can be present.
    with pytest.raises(ValueError):
        f(vs, vs)


def test_parametrised_double_initial_value(vs):
    @parametrised
    def f(vs_, x: Unbounded(init=5) = 5):
        pass

    with pytest.raises(ValueError):
        f(vs)


def test_parametrised_double_name(vs):
    @parametrised
    def f(vs_, x: Unbounded(name="x")):
        pass

    with pytest.raises(ValueError):
        f(vs)


def test_parametrised_unbounded(vs):
    @parametrised
    def f(vs_, x: Unbounded = 1):
        return x

    assert f(vs) == 1


def test_parametrised_positive(vs):
    @parametrised
    def f(vs_, x: Positive):
        return x

    assert f(vs) > 0


def test_parametrised_bounded(vs):
    @parametrised
    def f(vs_, x: Bounded(lower=10, upper=11)):
        return x

    assert 10 <= f(vs) <= 11


def test_parametrised_lower_triangular(vs):
    @parametrised
    def f(vs_, x: LowerTriangular(shape=(5, 5))):
        return x

    assert_lower_triangular(f(vs))


def test_parametrised_positive_definite(vs):
    @parametrised
    def f(vs_, x: PositiveDefinite(shape=(5, 5))):
        return x

    assert_positive_definite(f(vs))


def test_parametrised_orthogonal(vs):
    @parametrised
    def f(vs_, x: Orthogonal(shape=(5, 5))):
        return x

    assert_orthogonal(f(vs))
