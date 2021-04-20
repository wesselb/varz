import logging
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from functools import reduce
from operator import mul

import lab as B
import numpy as np
import wbml.out
from plum import Dispatcher

from .util import Packer, match, lazy_tf as tf, lazy_torch as torch, lazy_jnp as jnp

__all__ = ["Provider", "Vars"]

log = logging.getLogger(__name__)

_dispatch = Dispatcher()


@_dispatch
def _assign(x: B.NPNumeric, value: B.Numeric):
    np.copyto(x, value)
    return x


@_dispatch
def _assign(x: B.TFNumeric, value: B.Numeric):
    return x.assign(value)


@_dispatch
def _assign(x: B.TorchNumeric, value: B.Numeric):
    if not isinstance(value, B.TorchNumeric):
        value = torch.tensor(value, dtype=x.dtype)
    x.data.copy_(value)
    return x


@_dispatch
def _assign(x: B.JAXNumeric, value: B.Numeric):
    return jnp.array(value, dtype=x.dtype)


class Provider(metaclass=ABCMeta):
    @abstractmethod
    def unbounded(
        self, init=None, shape=None, dtype=None, name=None
    ):  # pragma: no cover
        """Get an unbounded variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (tuple[int], optional): Shape of the variable. Defaults to scalar.
            dtype (data type, optional): Data type of the variable. Defaults to that
                of the storage.
            name (str, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

    def get(self, *args, **kw_args):
        """Alias for :meth:`.vars.Provider.unbounded`."""
        return self.unbounded(*args, **kw_args)

    @abstractmethod
    def positive(
        self, init=None, shape=None, dtype=None, name=None
    ):  # pragma: no cover
        """Get a positive variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (tuple[int], optional): Shape of the variable. Defaults to scalar.
            dtype (data type, optional): Data type of the variable. Defaults to that
                of the storage.
            name (str, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

    def pos(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.positive`."""
        return self.positive(*args, **kw_args)

    @abstractmethod
    def bounded(
        self, init=None, lower=1e-4, upper=1e4, shape=None, dtype=None, name=None
    ):  # pragma: no cover
        """Get a bounded variable.

        Args:
            init (tensor, optional): Initialisation of the variable.
            lower (tensor, optional): Lower bound. Defaults to `1e-4`.
            upper (tensor, optional): Upper bound. Defaults to `1e4`.
            shape (tuple[int], optional): Shape of the variable. Defaults to scalar.
            dtype (data type, optional): Data type of the variable. Defaults to that
                of the storage.
            name (hashable, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

    def bnd(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.bounded`."""
        return self.bounded(*args, **kw_args)

    @abstractmethod
    def lower_triangular(
        self, init=None, shape=None, dtype=None, name=None
    ):  # pragma: no cover
        """Get a lower-triangular matrix.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (int, optional): Number of rows and columns of the matrix.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

    def tril(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.lower_triangular`."""
        return self.lower_triangular(*args, **kw_args)

    @abstractmethod
    def positive_definite(
        self, init=None, shape=None, dtype=None, name=None
    ):  # pragma: no cover
        """Get a positive-definite matrix.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (int, optional): Number of rows and columns of the matrix.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.

        Returns:
            tensor: Variable.
        """

    def pd(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.positive_definite`."""
        return self.positive_definite(*args, **kw_args)

    @abstractmethod
    def orthogonal(
        self, init=None, shape=None, dtype=None, name=None, method="svd"
    ):  # pragma: no cover
        """Get an orthogonal matrix.

        Args:
            init (tensor, optional): Initialisation of the variable.
            shape (int, optional): Number of rows and columns of the matrix.
            dtype (data type, optional): Data type of the variable. Defaults to
                that of the storage.
            name (hashable, optional): Name of the variable.
            method ('svd', 'expm' or 'cayley'): Parametrisation. Method of
                parametrisation. Defaults to 'svd'.

        Returns:
            tensor: Variable.
        """

    def orth(self, *args, **kw_args):
        """Alias for :meth:`.vars.Vars.orthogonal`."""
        return self.orthogonal(*args, **kw_args)

    @abstractmethod
    def __getitem__(self, name):  # pragma: no cover
        """Get a variable by name.

        Args:
            name (hashable): Name of variable.

        Returns:
            tensor: Variable.
        """


@_dispatch
def _check_matrix_shape(shape, square=True):
    raise ValueError(f"Object {shape} is not a shape.")


@_dispatch
def _check_matrix_shape(shape: tuple, square=True):
    if len(shape) != 2:
        raise ValueError(f"Shape {shape} must be the shape of a matrix.")
    if square and shape[0] != shape[1]:
        raise ValueError(f"Shape {shape} must be square.")


def _check_init_shape(init, shape):
    if init is None and shape is None:
        raise ValueError(
            f"The shape must be given to automatically initialise "
            f"a matrix variable."
        )
    if shape is None:
        shape = B.shape(init)
    return init, shape


class Vars(Provider):
    """Variable storage.

    Args:
        dtype (data type): Data type of the variables.
        source (tensor, optional): Tensor to source variables from. Defaults to
            not being used.
    """

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

    def _resolve_dtype(self, dtype):
        if dtype is None:
            return self.dtype
        else:
            return dtype

    def _get_var(
        self,
        transform,
        inverse_transform,
        init,
        generate_init,
        shape,
        shape_latent,
        dtype,
        name,
    ):
        # If the name already exists, return that variable.
        try:
            return self[name]
        except KeyError:
            pass

        # A new variable will be added. Clear lookup cache.
        self._get_vars_cache.clear()

        # Resolve data type.
        dtype = self._resolve_dtype(dtype)

        # If no source is provided, get the latent from from the provided
        # initialiser.
        if self.source is None:
            # Resolve initialisation and inverse transform.
            if init is None:
                init = generate_init(shape=shape, dtype=dtype)
            else:
                init = B.cast(dtype, init)
                if shape is not None and shape != B.shape(init):
                    raise ValueError(
                        f"Shape of initial value {B.shape(init)} is not equal to the "
                        f"desired shape {shape}."
                    )

            # Construct optimisable variable.
            latent = inverse_transform(init)
            if isinstance(self.dtype, B.TFDType):
                latent = tf.Variable(latent)
            elif isinstance(self.dtype, B.TorchDType):
                pass  # All is good in this case.
            elif isinstance(self.dtype, B.JAXDType):
                latent = jnp.array(latent)
            else:
                # Must be a NumPy data type.
                assert isinstance(self.dtype, B.NPDType)
                latent = np.array(latent)
        else:
            # Get the latent variable from the source.
            length = reduce(mul, shape_latent, 1)
            latent_flat = self.source[self.source_index : self.source_index + length]
            self.source_index += length

            # Cast to the right data type.
            latent = B.cast(dtype, B.reshape(latent_flat, *shape_latent))

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

    def unbounded(self, init=None, shape=None, dtype=None, name=None):
        # If nothing is specific, generate a scalar.
        if init is None and shape is None:
            shape = ()

        def generate_init(shape, dtype):
            return B.randn(dtype, *shape)

        return self._get_var(
            transform=lambda x: x,
            inverse_transform=lambda x: x,
            init=init,
            generate_init=generate_init,
            shape=shape,
            shape_latent=shape,
            dtype=dtype,
            name=name,
        )

    def positive(self, init=None, shape=None, dtype=None, name=None):
        # If nothing is specific, generate a scalar.
        if init is None and shape is None:
            shape = ()

        def generate_init(shape, dtype):
            return B.rand(dtype, *shape)

        return self._get_var(
            transform=lambda x: B.exp(x),
            inverse_transform=lambda x: B.log(x),
            init=init,
            generate_init=generate_init,
            shape=shape,
            shape_latent=shape,
            dtype=dtype,
            name=name,
        )

    def bounded(
        self, init=None, lower=1e-4, upper=1e4, shape=None, dtype=None, name=None
    ):
        # If nothing is specific, generate a scalar.
        if init is None and shape is None:
            shape = ()

        def transform(x):
            return lower + (upper - lower) / (1 + B.exp(-x))

        def inverse_transform(x):
            return B.log(x - lower) - B.log(upper - x)

        def generate_init(shape, dtype):
            return lower + B.rand(dtype, *shape) * (upper - lower)

        return self._get_var(
            transform=transform,
            inverse_transform=inverse_transform,
            init=init,
            generate_init=generate_init,
            shape=shape,
            shape_latent=shape,
            dtype=dtype,
            name=name,
        )

    def lower_triangular(self, init=None, shape=None, dtype=None, name=None):
        init, shape = _check_init_shape(init, shape)
        _check_matrix_shape(shape)

        # Result must be square. Get a side.
        side = shape[0]

        def transform(x):
            return B.vec_to_tril(x)

        def inverse_transform(x):
            return B.tril_to_vec(x)

        def generate_init(shape, dtype):
            mat = B.randn(dtype, *shape)
            return transform(B.tril_to_vec(mat))

        shape_latent = (int(side * (side + 1) / 2),)
        return self._get_var(
            transform=transform,
            inverse_transform=inverse_transform,
            init=init,
            generate_init=generate_init,
            shape=shape,
            shape_latent=shape_latent,
            dtype=dtype,
            name=name,
        )

    def positive_definite(self, init=None, shape=None, dtype=None, name=None):
        init, shape = _check_init_shape(init, shape)
        _check_matrix_shape(shape)

        # Result must be square. Get a side.
        side = shape[0]

        def transform(x):
            log_diag = x[:side]
            chol = B.vec_to_tril(x[side:], offset=-1) + B.diag(B.exp(log_diag))
            return B.matmul(chol, chol, tr_b=True)

        def inverse_transform(x):
            chol = B.cholesky(B.reg(x))
            return B.concat(B.log(B.diag(chol)), B.tril_to_vec(chol, offset=-1))

        def generate_init(shape, dtype):
            mat = B.randn(dtype, *shape)
            return B.matmul(mat, mat, tr_b=True)

        shape_latent = (int(side * (side + 1) / 2),)
        return self._get_var(
            transform=transform,
            inverse_transform=inverse_transform,
            init=init,
            generate_init=generate_init,
            shape=shape,
            shape_latent=shape_latent,
            dtype=dtype,
            name=name,
        )

    def orthogonal(self, init=None, shape=None, dtype=None, name=None, method="svd"):
        init, shape = _check_init_shape(init, shape)

        if method == "svd":
            _check_matrix_shape(shape, square=False)
            n, m = shape
            shape_latent = (n, m)

            # Fix singular values.
            sing_vals = B.linspace(self._resolve_dtype(dtype), 1, 2, min(n, m))

            def transform(x):
                u, s, v = B.svd(x)
                # u * v' is the closest orthogonal matrix to x in Frobenius norm.
                return B.matmul(u, v, tr_b=True)

            def inverse_transform(x):
                if n >= m:
                    return x * sing_vals[None, :]
                else:
                    return x * sing_vals[:, None]

            def generate_init(shape, dtype):
                mat = B.randn(dtype, *shape)
                return transform(mat)

        elif method == "expm":
            _check_matrix_shape(shape)
            side = shape[0]
            shape_latent = (int(side * (side + 1) / 2 - side),)

            def transform(x):
                tril = B.vec_to_tril(x, offset=-1)
                skew = tril - B.transpose(tril)
                return B.expm(skew)

            def inverse_transform(x):
                return B.tril_to_vec(B.logm(x), offset=-1)

            def generate_init(shape, dtype):
                mat = B.randn(dtype, *shape)
                return transform(B.tril_to_vec(mat, offset=-1))

        elif method == "cayley":
            _check_matrix_shape(shape)
            side = shape[0]
            shape_latent = (int(side * (side + 1) / 2 - side),)

            def transform(x):
                tril = B.vec_to_tril(x, offset=-1)
                skew = tril - B.transpose(tril)
                eye = B.eye(skew)
                return B.solve(eye + skew, eye - skew)

            def inverse_transform(x):
                eye = B.eye(x)
                skew = B.solve(eye + x, eye - x)
                return B.tril_to_vec(skew, offset=-1)

            def generate_init(shape, dtype):
                mat = B.randn(dtype, *shape)
                return transform(B.tril_to_vec(mat, offset=-1))

        else:
            raise ValueError(f'Unknown parametrisation "{method}".')

        return self._get_var(
            transform=transform,
            inverse_transform=inverse_transform,
            init=init,
            generate_init=generate_init,
            shape=shape,
            shape_latent=shape_latent,
            dtype=dtype,
            name=name,
        )

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
            # Do a differentiable assignment, but ensure that the data type is
            # right.
            dtype = B.dtype(self.vars[index])
            self.vars[index] = B.cast(dtype, value)
            return value
        else:
            # Overwrite data.
            self.vars[index] = _assign(
                self.vars[index], self.inverse_transforms[index](value)
            )
            return self.vars[index]

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

    def get_vars(self, *names, return_indices=False):
        """Get latent variables.

        If no arguments are supplied, then all latent variables are retrieved.
        Furthermore, the same collection of variables is guaranteed to be
        returned in the same order.

        Args:
            *names (hashable): Get variables by name.
            return_indices (bool, optional): Get the indices of the variables instead.
                Defaults to `False`.

        Returns:
            list: Matched latent variables or their indices, depending on the
                value of `indices`.
        """
        # If nothing is specified, return all latent variables.
        if len(names) == 0:
            if return_indices:
                return list(range(len(self.vars)))
            else:
                return self.vars

        # Attempt to use cache.
        try:
            indices = self._get_vars_cache[names]
        except KeyError:
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
                    raise ValueError(f'No variable matching "{name}".')

            # Sort the indices for a consistent result.
            indices = sorted(indices)

            # Store in cache before proceeding.
            self._get_vars_cache[names] = indices

        # Return indices if asked for. Otherwise, return variables.
        if return_indices:
            return indices
        else:
            return [self.vars[i] for i in indices]

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

    def set_vector(self, values, *names, differentiable=False):
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

        if differentiable:
            # Do a differentiable assignment.
            for index, value in zip(self.get_vars(*names, return_indices=True), values):
                self.vars[index] = value
            return values
        else:
            # Overwrite data.
            assignments = []
            for index, value in zip(self.get_vars(*names, return_indices=True), values):
                self.vars[index] = _assign(self.vars[index], value)
                assignments.append(self.vars[index])
            return assignments

    @property
    def names(self):
        """All available names."""
        return list(self.name_to_index.keys())

    def print(self):
        """Print all variables."""
        for name in self.names:
            wbml.out.kv(name, self[name])