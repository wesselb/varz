# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from lab import B
from plum import Dispatcher, Self, Referentiable

from varz.util import Packer

__all__ = ['Vars', 'vars64', 'vars32']

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


class Vars(Referentiable):
    """Variable storage manager.

    Args:
        dtype (data type, optional): Data type of the variables. Defaults to
            `np.float32`.
    """
    _dispatch = Dispatcher(in_class=Self)

    def __init__(self, dtype=np.float64):
        self.dtype = dtype

        # Storage:
        self.vars = []
        self.transforms = []
        self.inverse_transforms = []

        # Lookup:
        self.names = {}
        self.groups = {}

        # Packing:
        self.vector_packer = None

    def detach(self):
        """Create a detached copy of the variable manager."""
        vs = Vars(dtype=self.dtype)
        vs.transforms = self.transforms
        vs.inverse_transforms = self.inverse_transforms
        vs.names = self.names
        vs.groups = self.groups
        for var in self.vars:
            vs.vars.append(B.Variable(var.clone().detach_()))
        return vs

    def detach_vars(self):
        """Detached the variables held."""
        self.vars = [B.Variable(var.detach_()) for var in self.vars]

    def requires_grad(self, value, *names, **kw_args):
        for var in self.get_vars(*names, **kw_args):
            var.requires_grad_(value)

    def get_vars(self, *names, **kw_args):
        """Get latent variables.

        If no arguments are supplied, then all latent variables are retrieved.
        Furthermore, the same collection of variables is guaranteed to be
        returned in the same order.

        Args:
            *names (hashable): Get variables by name.
            groups (list[hashable]): Get variables by group.

        Returns:
            list[tensor]: Matched latent variables.
        """
        groups = kw_args['groups'] if 'groups' in kw_args else None

        # If nothing is specified, return all latent variables.
        if len(names) == 0 and not groups:
            return self.vars

        # Collect indices of matches.
        indices = set()

        # Collect by name.
        if names:
            indices |= {self.names[name] for name in names}

        # Collect by group.
        if groups:
            for group in groups:
                indices |= set(self.groups[group])

        # Collect variables and return.
        return [self.vars[i] for i in sorted(indices)]

    def get_vector(self, *names, **kw_args):
        """Get all the latent variables stacked in a vector.

        If no arguments are supplied, then all latent variables are retrieved.

        Args:
            *names (hashable): Get variables by name.
            groups (list[hashable]): Get variables by group.

        Returns:
            tensor: Vector consisting of all latent values
        """
        vars = self.get_vars(*names, **kw_args)
        self.vector_packer = Packer(*vars)
        return self.vector_packer.pack(*vars)

    def set_vector(self, values, *names, **kw_args):
        """Set all the latent variables by values from a vector.

        If no arguments are supplied, then all latent variables are retrieved.

        Args:
            values (tensor): Vector to set the variables to.
            *names (hashable): Set variables by name.
            groups (list[hashable]): Set variables by group.

        Returns:
            list: Assignment results.
        """
        vars = self.get_vars(*names, **kw_args)
        values = self.vector_packer.unpack(values)
        assignments = []
        for var, value in zip(vars, values):
            assignments.append(B.assign(var, value))
        return assignments

    def init(self, session):
        """Initialise the variables.

        Args:
            session (:class:`B.Session`): TensorFlow session.
        """
        session.run(B.variables_initializer(self.vars))

    def get(self, init=None, shape=(), dtype=None, name=None, group=None):
        """Get an unbounded variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.
            group (hashable, optional): Group of the variable.

        Returns:
            tensor: Variable.
        """

        def generate_init(shape, dtype):
            return B.randn(shape, dtype=dtype)

        return self._get_var(transform=lambda x: x,
                             inverse_transform=lambda x: x,
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name,
                             group=group)

    def positive(self, init=None, shape=(), dtype=None, name=None, group=None):
        """Get a positive variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (tuple[int], optional): Shape of the variable. Defaults to
                scalar.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.
            group (hashable, optional): Group of the variable.

        Returns:
            tensor: Variable.
        """

        def generate_init(shape, dtype):
            return B.rand(shape, dtype=dtype)

        return self._get_var(transform=lambda x: B.exp(x),
                             inverse_transform=lambda x: B.log(x),
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name,
                             group=group)

    def pos(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.positive`."""
        return self.positive(*args, **kw_args)

    def bounded(self,
                init=None,
                lower=1e-4,
                upper=1e4,
                shape=(),
                dtype=None,
                name=None,
                group=None):
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
            group (hashable, optional): Group of the variable.

        Returns:
            tensor: Variable.
        """

        def transform(x):
            return lower + (upper - lower) / (1 + B.exp(x))

        def inverse_transform(x):
            return B.log(upper - x) - B.log(x - lower)

        def generate_init(shape, dtype):
            return lower + B.rand(shape, dtype=dtype) * (upper - lower)

        return self._get_var(transform=transform,
                             inverse_transform=inverse_transform,
                             init=init,
                             generate_init=generate_init,
                             shape=shape,
                             dtype=dtype,
                             name=name,
                             group=group)

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
                 name,
                 group):
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
            init = B.array(init, dtype=dtype)

        # Construct latent variable and store transforms.
        latent = B.Variable(inverse_transform(init))
        self.vars.append(latent)
        self.transforms.append(transform)
        self.inverse_transforms.append(inverse_transform)

        # Get index of the variable.
        index = len(self.vars) - 1

        # Store name if given.
        if name is not None:
            self.names[name] = index

        # Store group if given.
        if group is not None:
            try:
                self.groups[group].append(index)
            except KeyError:
                self.groups[group] = [index]

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
        index = self.names[name]
        return B.assign(self.vars[index], self.inverse_transforms[index](value))

    def __getitem__(self, name):
        """Get a variable by name.

        Args:
            name (hashable): Name of variable.

        Returns:
            tensor: Variable.
        """
        index = self.names[name]
        return self.transforms[index](self.vars[index])


vars32 = Vars(np.float32)
vars64 = Vars(np.float64)
