# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

from itertools import product

import numpy as np
import lab as B

from . import Vars, Initialiser, Packer
# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, allclose


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
        yield eq, vs['a'], values[0]
        yield allclose, vs['b'], values[1]


def test_packer():
    a, b, c = B.randn(5, 10), B.randn(20), B.randn(5, 1, 15)

    for packer, args in zip([Packer(a, b, c), Packer([a, b, c])],
                            [(a, b, c), ((a, b, c),)]):
        # Test packing.
        packed = packer.pack(*args)
        yield eq, B.rank(packed), 1

        # Test unpacking.
        a_, b_, c_ = packer.unpack(packed)
        yield allclose, a, a_
        yield allclose, b, b_
        yield allclose, c, c_
