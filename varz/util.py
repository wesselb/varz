# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging
import re
from itertools import product

import lab as B
from plum import Referentiable, Dispatcher, Self

__all__ = ['Initialiser', 'Packer', 'match']

log = logging.getLogger(__name__)


class Initialiser(object):
    """Variable initialiser."""

    def __init__(self):
        self._assignments = {}

    def assign(self, name, values):
        """Assign values to a particular variable.

        Args:
            name (str): Name of variables.
            values (list[tensor]): List of values to assign.
        """
        self._assignments[name] = values

    def generate(self, vs):
        """Generate initialisers.

        Args:
            vs (:class:`.vars.Vars`): Variable storage.

        Returns:
            list[function]: List of functions that perform the initialisations.
        """
        names, value_sets = zip(*self._assignments.items())
        return [_construct_assigner(vs, names, values)
                for values in product(*value_sets)]


def _construct_assigner(vs, names, values):
    def assign():
        return [vs.assign(name, val) for name, val in zip(names, values)]

    return assign


class Packer(Referentiable):
    """Pack objects into a vector.

    Args:
        *objs (tensor): Objects to pack.
    """
    _dispatch = Dispatcher(in_class=Self)

    @_dispatch([object])
    def __init__(self, *objs):
        self._shapes = [B.shape(obj) for obj in objs]
        self._lengths = [B.length(obj) for obj in objs]

    @_dispatch({tuple, list})
    def __init__(self, objs):
        Packer.__init__(self, *objs)

    @_dispatch([object])
    def pack(self, *objs):
        """Pack objects.

        Args:
            *objs (tensor): Objects to pack.

        Returns:
            tensor: Vector representation of the objects.
        """
        return B.concat([B.reshape(obj, shape=[-1]) for obj in objs], axis=0)

    @_dispatch({tuple, list})
    def pack(self, objs):
        return self.pack(*objs)

    def unpack(self, package):
        """Unpack vector.

        Args:
            package (tensor): Vector to unpack.

        Returns:
            list[tensor]: Original objects.
        """
        i, outs = 0, []
        for shape, length in zip(self._shapes, self._lengths):
            outs.append(B.reshape(package[i:i + length], shape=shape))
            i += length
        return outs


def match(pattern, target):
    """Match a string exactly against a pattern, where the pattern is a regex
    with '*' as the only active character.

    Args:
        pattern (str): Pattern.
        target (str): Target.

    bool: `True` if `pattern` matches `target`.
    """
    pattern = ''.join('.*' if c == '*' else re.escape(c) for c in pattern)
    return bool(re.match('^' + pattern + '$', target))
