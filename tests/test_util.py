import lab as B
import pytest

from varz import pack, unpack
from .util import approx


def test_pack_unpack():
    a, b, c = B.randn(5, 10), B.randn(20), B.randn(5, 1, 15)

    # Test packing.
    package = pack(a, b, c)
    assert B.rank(package) == 1

    # Test unpacking.
    a2, b2, c2 = unpack(package, B.shape(a), B.shape(b), B.shape(c))
    approx(a, a2)
    approx(b, b2)
    approx(c, c2)

    # Check that the package must be a vector.
    with pytest.raises(ValueError):
        unpack(B.randn(2, 2), (2, 2))
