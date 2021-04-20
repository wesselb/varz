from itertools import product

import lab as B
import numpy as np
from varz import Vars, Initialiser, Packer

from .util import approx, vs


def test_initialiser(vs):
    init = Initialiser()

    # Initialise two variables.
    vs.get(1.0, name="a")
    vs.pos(2.0, name="b")

    # Define some initialisations.
    init.assign("a", [-3.0, 4.0])
    init.assign("b", [5.0, 6.0])
    inits = init.generate(vs)

    # Test the initialisations.
    for initialiser, values in zip(inits, product([-3.0, 4.0], [5.0, 6.0])):
        initialiser()
        assert vs["a"] == values[0]
        approx(vs["b"], values[1])


def test_packer():
    a, b, c = B.randn(5, 10), B.randn(20), B.randn(5, 1, 15)

    for packer, args in zip(
        [Packer(a, b, c), Packer([a, b, c])], [(a, b, c), ((a, b, c),)]
    ):
        # Test packing.
        packed = packer.pack(*args)
        assert B.rank(packed) == 1

        # Test unpacking.
        a_, b_, c_ = packer.unpack(packed)
        approx(a, a_)
        approx(b, b_)
        approx(c, c_)
