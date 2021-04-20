import importlib
import logging
import re
from itertools import product
from typing import Union

import lab as B
from plum import Dispatcher

__all__ = ["lazy_tf", "lazy_torch", "lazy_jnp", "Initialiser", "Packer", "match"]

_dispatch = Dispatcher()

log = logging.getLogger(__name__)


class LazyModule:
    """A module that loads once an attribute is requested.

    Args:
        name (str): Name of module.
    """

    def __init__(self, name):
        self.name = name
        self.module = None

    def __getattr__(self, item):
        # Load module.
        if self.module is None:
            self.module = importlib.import_module(self.name)

        return getattr(self.module, item)


lazy_tf = LazyModule("tensorflow")
lazy_torch = LazyModule("torch")
lazy_jnp = LazyModule("jax.numpy")


class Initialiser:
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
        return [
            _construct_assigner(vs, names, values) for values in product(*value_sets)
        ]


def _construct_assigner(vs, names, values):
    def assign():
        return [vs.assign(name, val) for name, val in zip(names, values)]

    return assign


class Packer:
    """Pack objects into a vector.

    Args:
        *objs (tensor): Objects to pack.
    """

    @_dispatch
    def __init__(self, *objs):
        self._shapes = [B.shape(obj) for obj in objs]
        self._lengths = [B.length(obj) for obj in objs]

    @_dispatch
    def __init__(self, objs: Union[tuple, list]):
        Packer.__init__(self, *objs)

    @_dispatch
    def pack(self, *objs):
        """Pack objects.

        Args:
            *objs (tensor): Objects to pack.

        Returns:
            tensor: Vector representation of the objects.
        """
        return B.concat(*[B.flatten(obj) for obj in objs], axis=0)

    @_dispatch
    def pack(self, objs: Union[tuple, list]):
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
            outs.append(B.reshape(package[i : i + length], *shape))
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
    pattern = "".join(".*" if c == "*" else re.escape(c) for c in pattern)
    return bool(re.match("^" + pattern + "$", target))
