import importlib
import logging
import re
from itertools import product
from typing import Union
from functools import reduce
from operator import mul

import lab as B
from plum import Dispatcher

__all__ = ["lazy_tf", "lazy_torch", "lazy_jnp", "pack", "unpack", "match"]

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


@_dispatch
def pack(*objs: B.Numeric):
    """Pack objects.

    Args:
        *objs (tensor): Objects to pack.

    Returns:
        tensor: Vector representation of the objects.
    """
    return B.concat(*[B.flatten(obj) for obj in objs], axis=0)


@_dispatch
def unpack(package: B.Numeric, *shapes):
    """Unpack vector.

    Args:
        package (tensor): Tensor to unpack.
        *shapes (shape): Shapes of objects to unpack.

    Returns:
        list[tensor]: Original objects.
    """
    if B.rank(package) != 1:
        raise ValueError("Package must be a vector.")

    # Unpack package.
    lengths = [reduce(mul, shape, 1) for shape in shapes]
    i, outs = 0, []
    for length, shape in zip(lengths, shapes):
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
