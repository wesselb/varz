# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

# noinspection PyUnresolvedReferences
from . import eq, neq, lt, le, ge, gt, raises, call, ok, lam


def test():
    yield eq, 1, 1