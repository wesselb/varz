# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import product

import lab as B
import numpy as np

from varz import Vars, Initialiser, Packer
from .util import allclose


def test_initialiser():
    vs, init = Vars(np.float64), Initialiser()

    # Initialise two variables.
    vs.get(1., name='a')
    vs.pos(2., name='b')

    # Define some initialisations.
    init.assign('a', [-3., 4.])
    init.assign('b', [5., 6.])
    inits = init.generate(vs)

    # Test the initialisations.
    for initialiser, values in zip(inits, product([-3., 4.], [5., 6.])):
        initialiser()
        assert vs['a'] == values[0]
        allclose(vs['b'], values[1])


def test_packer():
    a, b, c = B.randn(5, 10), B.randn(20), B.randn(5, 1, 15)

    for packer, args in zip([Packer(a, b, c), Packer([a, b, c])],
                            [(a, b, c), ((a, b, c),)]):
        # Test packing.
        packed = packer.pack(*args)
        assert B.rank(packed) == 1

        # Test unpacking.
        a_, b_, c_ = packer.unpack(packed)
        allclose(a, a_)
        allclose(b, b_)
        allclose(c, c_)
