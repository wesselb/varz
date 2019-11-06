import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import reduce
from operator import mul

import lab as B
import numpy as np
import wbml.out
from plum import Dispatcher, Self, Referentiable

from .util import Packer, match, lazy_tf as tf, lazy_torch as torch

__all__ = ['Provider', 'Vars']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


@_dispatch(B.NPNumeric, B.Numeric)
def _assign(x, value):
    np.copyto(x, value)
    return x


@_dispatch(B.TFNumeric, B.Numeric)
def _assign(x, value):
    return x.assign(value)


@_dispatch(B.TorchNumeric, B.Numeric)
def _assign(x, value):
    if not isinstance(value, B.TorchNumeric):
        value = torch.tensor(value, dtype=x.dtype)
    x.data.copy_(value)
    return x


class Provider(metaclass=Referentiable(ABCMeta)):
    @abstractmethod
    def unbounded(self,
                  init=None,
                  shape=(),
                  dtype=None,
                  name=None):  # pragma: no cover
        """Get an unbounded variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (str, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

    def get(self, *args, **kw_args):
        """Alias for :meth:`.vars.Provider.unbounded`."""
        return self.unbounded(*args, **kw_args)

    @abstractmethod
    def positive(self,
                 init=None,
                 shape=(),
                 dtype=None,
                 name=None):  # pragma: no cover
        """Get a positive variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (str, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

    def pos(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.positive`."""
        return self.positive(*args, **kw_args)

    @abstractmethod
    def bounded(self,
                init=None,
                lower=1e-4,
                upper=1e4,
                shape=(),
                dtype=None,
                name=None):  # pragma: no cover
        """Get a bounded variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            lower (tensor, optional): Lower bound. Defaults to `1e-4`.
            upper (tensor, optional): Upper bound. Defaults to `1e4`.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

    def bnd(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.bounded`."""
        return self.bounded(*args, **kw_args)

    @abstractmethod
    def __getitem__(self, name):  # pragma: no cover
        """Get a variable by name.

        Args:
            name (hashable): Name of variable.

        Returns:
            tensor: Variable.
        """


class Vars(Provider):
    """Variable storage.

    Args:
        dtype (data type): Data type of the variables.
        source (tensor, optional): Tensor to source variables from. Defaults to
            not being used.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, dtype, source=None):
        self.dtype = dtype

        # Source:
        self.source = source
        self.source_index = 0

        # Storage:
        self.vars = []
        self.transforms = []
        self.inverse_transforms = []

        # Lookup:
        self.name_to_index = OrderedDict()
        self._get_vars_cache = {}

        # Packing:
        self.vector_packer = None

    def _get_var(self,
                 transform,
                 inverse_transform,
                 init,
                 generate_init,
                 shape,
                 dtype,
                 name):
        # If the name already exists, return that variable.
        try:
            return self[name]
        except KeyError:
            pass

        # A new variable will be added. Clear lookup cache.
        self._get_vars_cache.clear()

        # Resolve data type.
        dtype = self.dtype if dtype is None else dtype

        # If no source is provided, get the latent from from the provided
        # initialiser.
        if self.source is None:
            # Resolve initialisation and inverse transform.
            if init is None:
                init = generate_init(shape=shape, dtype=dtype)
            else:
                init = B.cast(dtype, init)

            # Construct optimisable variable.
            latent = inverse_transform(init)
            if isinstance(self.dtype, B.TFDType):
                latent = tf.Variable(latent)
            elif isinstance(self.dtype, B.TorchDType):
                pass  # All is good in this case.
            else:
                # Must be a NumPy data type.
                assert isinstance(self.dtype, B.NPDType)
                latent = np.array(latent)
        else:
            # Get the latent variable from the source.
            length = reduce(mul, shape, 1)
            latent_flat = \
                self.source[self.source_index:self.source_index + length]
            self.source_index += length

            # Cast to the right data type.
            latent = B.cast(dtype, B.reshape(latent_flat, *shape))

        # Store transforms.
        self.vars.append(latent)
        self.transforms.append(transform)
        self.inverse_transforms.append(inverse_transform)

        # Get index of the variable.
        index = len(self.vars) - 1

        # Store name if given.
        if name is not None:
            self.name_to_index[name] = index

        # Generate the variable and return.
        return transform(latent)

    def unbounded(self, init=None, shape=(), dtype=None, name=None):
        def generate_init(shape, dtype):
            return B.randn(dtype, *shape)

        return self._get_var(transform=lambda x: x,
                             inverse_transform=lambda x: x,
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def positive(self, init=None, shape=(), dtype=None, name=None):
        def generate_init(shape, dtype):
            return B.rand(dtype, *shape)

        return self._get_var(transform=lambda x: B.exp(x),
                             inverse_transform=lambda x: B.log(x),
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def bounded(self,
                init=None,
                lower=1e-4,
                upper=1e4,
                shape=(),
                dtype=None,
                name=None):
        def transform(x):
            return lower + (upper - lower) / (1 + B.exp(x))

        def inverse_transform(x):
            return B.log(upper - x) - B.log(x - lower)

        def generate_init(shape, dtype):
            return lower + B.rand(dtype, *shape) * (upper - lower)

        return self._get_var(transform=transform,
                             inverse_transform=inverse_transform,
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def __getitem__(self, name):
        index = self.name_to_index[name]
        return self.transforms[index](self.vars[index])

    def assign(self, name, value, differentiable=False):
        """Assign a value to a variable.

        Args:
            name (hashable): Name of variable to assign value to.
            value (tensor): Value to assign.
            differentiable (bool, optional): Do a differentiable assignment.

        Returns:
            tensor: Assignment result.
        """
        index = self.name_to_index[name]
        if differentiable:
            # Do a differentiable assignment.
            self.vars[index] = value
            return value
        else:
            # Overwrite data.
            return _assign(self.vars[index],
                           self.inverse_transforms[index](value))

    def copy(self, detach=False):
        """Create a copy of the variable manager that shares the variables.

        Args:
            detach (bool, optional): Detach the variables in PyTorch. Defaults
                to `False`.

        Returns:
            :class:`.vars.Vars`: Copy.
        """
        vs = Vars(dtype=self.dtype)
        vs.transforms = list(self.transforms)
        vs.inverse_transforms = list(self.inverse_transforms)
        vs.name_to_index = OrderedDict(self.name_to_index)
        vs.vector_packer = self.vector_packer
        if detach:
            for var in self.vars:
                vs.vars.append(var.detach())
        else:
            vs.vars = list(self.vars)
        return vs

    def detach(self):
        """Detach all variables held in PyTorch."""
        self.vars = [v.detach() for v in self.vars]

    def requires_grad(self, value, *names):
        """Set which variables require a gradient in PyTorch.

        Args:
            value (bool): Require a gradient.
            *names (hashable): Specify variables by name.
        """
        for var in self.get_vars(*names):
            var.requires_grad_(value)

    def get_vars(self, *names, **kw_args):
        """Get latent variables.

        If no arguments are supplied, then all latent variables are retrieved.
        Furthermore, the same collection of variables is guaranteed to be
        returned in the same order.

        Args:
            *names (hashable): Get variables by name.
            indices (bool, optional): Get the indices of the variables instead.
                Defaults to `False`.

        Returns:
            list: Matched latent variables or their indices, depending on the
                value of `indices`.
        """
        # If nothing is specified, return all latent variables.
        if len(names) == 0:
            if kw_args.get('indices', False):
                return list(range(len(self.vars)))
            else:
                return self.vars

        # Attempt to use cache.
        cache_key = (names, kw_args.get('indices', False))
        try:
            return self._get_vars_cache[cache_key]
        except KeyError:
            pass

        # Collect indices of matches.
        indices = set()
        for name in names:
            a_match = False
            for k, v in self.name_to_index.items():
                if match(name, k):
                    indices |= {v}
                    a_match = True

            # Check that there was a match.
            if not a_match:
                raise ValueError('No variable matching "{}".'.format(name))

        # Return indices if asked for. Otherwise, return variables.
        if kw_args.get('indices', False):
            res = sorted(indices)
        else:
            res = [self.vars[i] for i in sorted(indices)]

        # Store in cache before returning.
        self._get_vars_cache[cache_key] = res
        return res

    def get_vector(self, *names):
        """Get all the latent variables stacked in a vector.

        If no arguments are supplied, then all latent variables are retrieved.

        Args:
            *names (hashable): Get variables by name.

        Returns:
            tensor: Vector consisting of all latent values
        """
        vars = self.get_vars(*names)
        self.vector_packer = Packer(*vars)
        return self.vector_packer.pack(*vars)

    def set_vector(self, values, *names, **kw_args):
        """Set all the latent variables by values from a vector.

        If no arguments are supplied, then all latent variables are retrieved.

        Args:
            values (tensor): Vector to set the variables to.
            *names (hashable): Set variables by name.
            differentiable (bool, optional): Differentiable assignment. Defaults
                to `False`.

        Returns:
            list: Assignment results.
        """
        values = self.vector_packer.unpack(values)

        if kw_args.get('differentiable', False):
            # Do a differentiable assignment.
            for index, value in zip(self.get_vars(*names, indices=True),
                                    values):
                self.vars[index] = value
            return values
        else:
            # Overwrite data.
            assignments = []
            for var, value in zip(self.get_vars(*names), values):
                assignments.append(_assign(var, value))
            return assignments

    @property
    def names(self):
        """All available names."""
        return list(self.name_to_index.keys())

    def print(self):
        """Print all variables."""
        for name in self.names:
            wbml.out.kv(name, self[name])
