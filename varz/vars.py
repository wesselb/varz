# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import lab as B
import numpy as np
import tensorflow as tf
from plum import Dispatcher, Self, Referentiable

from .util import Packer, match

__all__ = ['Vars']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


@_dispatch(B.NPNumeric, B.Numeric)
def _assign(x, value):
    np.copyto(x, value)
    return x


@_dispatch(B.TFNumeric, B.Numeric)
def _assign(x, value):
    return tf.assign(x, value)


@_dispatch(B.TorchNumeric, B.Numeric)
def _assign(x, value):
    x.data.copy_(value)
    return x


class Vars(Referentiable):
    """Variable storage manager.

    Args:
        dtype (data type): Data type of the variables.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, dtype):
        self.dtype = dtype

        # Storage:
        self.vars = []
        self.transforms = []
        self.inverse_transforms = []

        # Lookup:
        self.index_by_name = {}

        # Packing:
        self.vector_packer = None

    def detach(self):
        """Create a detached copy of the variable manager in PyTorch.

        Returns:
            :class:`.vars.Vars`: Detached copy.
        """
        vs = Vars(dtype=self.dtype)
        vs.transforms = self.transforms
        vs.inverse_transforms = self.inverse_transforms
        vs.index_by_name = self.index_by_name
        for var in self.vars:
            vs.vars.append(var.detach())
        return vs

    def detach_vars(self):
        """Detach all variables held in PyTorch."""
        for var in self.vars:
            var.detach_()

    def requires_grad(self, value, *names):
        """Set which variables require a gradient in PyTorch.

        Args:
            value (bool): Require a gradient.
            *names (hashable): Specify variables by name.
        """
        for var in self.get_vars(*names):
            var.requires_grad_(value)

    def get_vars(self, *names):
        """Get latent variables.

        If no arguments are supplied, then all latent variables are retrieved.
        Furthermore, the same collection of variables is guaranteed to be
        returned in the same order.

        Args:
            *names (hashable): Get variables by name.

        Returns:
            list[tensor]: Matched latent variables.
        """
        # If nothing is specified, return all latent variables.
        if len(names) == 0:
            return self.vars

        # Collect indices of matches.
        indices = set()
        for name in names:
            a_match = False
            for k, v in self.index_by_name.items():
                if match(name, k):
                    indices |= {v}
                    a_match = True

            # Check that there was a match.
            if not a_match:
                raise ValueError('No variable matching "{}".'.format(name))

        # Collect variables and return.
        return [self.vars[i] for i in sorted(indices)]

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

    def set_vector(self, values, *names):
        """Set all the latent variables by values from a vector.

        If no arguments are supplied, then all latent variables are retrieved.

        Args:
            values (tensor): Vector to set the variables to.
            *names (hashable): Set variables by name.

        Returns:
            list: Assignment results.
        """
        vars = self.get_vars(*names)
        values = self.vector_packer.unpack(values)
        assignments = []
        for var, value in zip(vars, values):
            assignments.append(_assign(var, value))
        return assignments

    def init(self, session):
        """Initialise the variables.

        Args:
            session (:class:`B.Session`): TensorFlow session.
        """
        session.run(tf.variables_initializer(self.vars))

    def get(self, init=None, shape=(), dtype=None, name=None):
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

        def generate_init(shape, dtype):
            return B.randn(shape, dtype)

        return self._get_var(transform=lambda x: x,
                             inverse_transform=lambda x: x,
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def positive(self, init=None, shape=(), dtype=None, name=None):
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

        def generate_init(shape, dtype):
            return B.rand(shape, dtype)

        return self._get_var(transform=lambda x: B.exp(x),
                             inverse_transform=lambda x: B.log(x),
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def pos(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.positive`."""
        return self.positive(*args, **kw_args)

    def bounded(self,
                init=None,
                lower=1e-4,
                upper=1e4,
                shape=(),
                dtype=None,
                name=None):
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

        def transform(x):
            return lower + (upper - lower) / (1 + B.exp(x))

        def inverse_transform(x):
            return B.log(upper - x) - B.log(x - lower)

        def generate_init(shape, dtype):
            return lower + B.rand(shape, dtype) * (upper - lower)

        return self._get_var(transform=transform,
                             inverse_transform=inverse_transform,
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name)

    def bnd(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.bounded`."""
        return self.bounded(*args, **kw_args)

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

        # Resolve data type.
        dtype = self.dtype if dtype is None else dtype

        # Resolve initialisation and inverse transform.
        if init is None:
            init = generate_init(shape=shape, dtype=dtype)
        else:
            init = B.cast(init, dtype)

        # Construct optimisable variable.
        latent = inverse_transform(init)
        if isinstance(self.dtype, B.TFDType):
            latent = tf.Variable(latent)
        elif isinstance(self.dtype, B.TorchDType):
            pass  # All is good in this case.
        else:
            # Must be a NumPy data type.
            latent = np.array(latent)

        # Store transforms.
        self.vars.append(latent)
        self.transforms.append(transform)
        self.inverse_transforms.append(inverse_transform)

        # Get index of the variable.
        index = len(self.vars) - 1

        # Store name if given.
        if name is not None:
            self.index_by_name[name] = index

        # Generate the variable and return.
        return transform(latent)

    def assign(self, name, value):
        """Assign a value to a variable.

        Args:
            name (hashable): Name of variable to assign value to.
            value (tensor): Value to assign.

        Returns:
            tensor: TensorFlow tensor that can be run to perform the assignment.
        """
        index = self.index_by_name[name]
        return _assign(self.vars[index], self.inverse_transforms[index](value))

    def __getitem__(self, name):
        """Get a variable by name.

        Args:
            name (hashable): Name of variable.

        Returns:
            tensor: Variable.
        """
        index = self.index_by_name[name]
        return self.transforms[index](self.vars[index])
