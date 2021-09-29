import abc
import inspect
from abc import ABCMeta, abstractmethod
from functools import wraps
from inspect import isfunction
from itertools import repeat

from .vars import Provider

__all__ = [
    "sequential",
    "namespace",
    "Struct",
    "Unbounded",
    "Positive",
    "Bounded",
    "LowerTriangular",
    "PositiveDefinite",
    "Orthogonal",
    "parametrised",
]


class _RedirectedProvided(Provider):
    """Redirect all methods of :class:`.vars.Provider` to a method `_get_var`.

    Args:
        vs (:class:`.vs.Vars`): Variable container to wrap.
    """

    def __init__(self, vs):
        self._vs = vs

    @abc.abstractmethod
    def _get_var(self, getter, args, kw_args):  # pragma: no cover
        pass

    def unbounded(self, *args, **kw_args):
        return self._get_var(self._vs.unbounded, args, kw_args)

    def positive(self, *args, **kw_args):
        return self._get_var(self._vs.positive, args, kw_args)

    def bounded(self, *args, **kw_args):
        return self._get_var(self._vs.bounded, args, kw_args)

    def lower_triangular(self, *args, **kw_args):
        return self._get_var(self._vs.lower_triangular, args, kw_args)

    def positive_definite(self, *args, **kw_args):
        return self._get_var(self._vs.positive_definite, args, kw_args)

    def orthogonal(self, *args, **kw_args):
        return self._get_var(self._vs.orthogonal, args, kw_args)

    def __getitem__(self, name):
        return self._vs[name]


class Sequential(_RedirectedProvided):
    """A variable provider that wraps a :class:`.vars.Vars` object and
    automatically names unnamed variables sequentially (0, 1, 2, ...) with a
    possible prefix which defaults to "var".

    Args:
        vs (:class:`.vs.Vars`): Variable container to wrap.
        prefix (str): Prefix for names in the sequence.
    """

    def __init__(self, vs, prefix):
        _RedirectedProvided.__init__(self, vs)
        self._prefix = prefix
        self._count = 0

    def _get_var(self, getter, args, kw_args):
        if "name" not in kw_args:
            kw_args["name"] = f"{self._prefix}{self._count}"
            self._count += 1
        return getter(*args, **kw_args)


def _to_sequential(x, prefix):
    """Convert a variable provider to sequential, if it is one.

    Args:
        x (object): Object to convert.
        prefix (str): Prefix for names in the sequence.

    Returns:
        object: `x` converted to sequential if it is a variable provider,
            otherwise just `x`.
    """
    return Sequential(x, prefix) if isinstance(x, Provider) else x


def _extract_prefix_and_f(prefix_or_f, default):
    """Extract the prefix and function.

    Args:
        prefix_or_f (object): Either a prefix or a function.
        default (str): Default prefix.

    Returns:
        tuple: Tuple containing the predict and the function. If a function
            was given, the prefix defaults to an empty string; and if a prefix
            was given, the function defaults to `None`.
    """
    if isfunction(prefix_or_f):
        # Used as a decorator.
        return default, prefix_or_f
    else:
        # Not used as decorator and prefix is specified.
        return prefix_or_f, None


def sequential(prefix="var"):
    """Decorator that generates variable names for unnamed variables
    sequentially.

    Args:
        prefix (str, optional): Prefix to prepend to the name of generated
            variables. Defaults to "var".
    """
    prefix, f = _extract_prefix_and_f(prefix, default="var")

    def decorator(f_):
        @wraps(f_)
        def wrapped_f(*args, **kw_args):
            # Replace all variable containers with their sequential variants.
            args = [_to_sequential(x, prefix) for x in args]
            kw_args = {k: _to_sequential(v, prefix) for k, v in kw_args.items()}

            return f_(*args, **kw_args)

        return wrapped_f

    # Return the decorated function directly if a function was given.
    return decorator if f is None else decorator(f)


class Namespace(_RedirectedProvided):
    """A variable provider that wraps a :class:`.vars.Vars` object and automatically
    prepends a prefix to named variables.

    Args:
        vs (:class:`.vs.Vars`): Variable container to wrap.
        prefix (str): Prefix for names.
    """

    def __init__(self, vs, prefix):
        _RedirectedProvided.__init__(self, vs)
        # Ensure that the prefix ends with a ".".
        if not prefix.endswith("."):
            prefix = prefix + "."
        self._prefix = prefix

    def _get_var(self, getter, args, kw_args):
        if "name" in kw_args:
            kw_args["name"] = self._prefix + kw_args["name"]
        return getter(*args, **kw_args)


def _to_namespace(x, prefix):
    """Convert a variable provider to a namespace, if it is one.

    Args:
        x (object): Object to convert.
        prefix (str): Prefix for names.

    Returns:
        object: `x` converted to a namespace if it is a variable provider,
            otherwise just `x`.
    """
    return Namespace(x, prefix) if isinstance(x, Provider) else x


def namespace(prefix):
    """Decorator that prefixes named variables.

    Args:
        prefix (str): Prefix to prepend to the name of named variables.
    """

    def decorator(f_):
        @wraps(f_)
        def wrapped_f(*args, **kw_args):
            # Replace all variable containers with their namespace variants.
            args = [_to_namespace(x, prefix) for x in args]
            kw_args = {k: _to_namespace(v, prefix) for k, v in kw_args.items()}

            return f_(*args, **kw_args)

        return wrapped_f

    return decorator


class Struct(_RedirectedProvided):
    """A variable provider that wraps a :class:`.vars.Vars` object and allows variables
    to be automatically named by getting attributes and indexing.

    Args:
        vs (:class:`.vs.Vars`): Variable container to wrap.
        path (str, optional): Path. Defaults to no path.
    """

    def __init__(self, vs, path=None):
        _RedirectedProvided.__init__(self, vs)
        self._path = path

    def _resolve_path(self, key, separator=""):
        if self._path:
            return f"{self._path}{separator}{key}"
        else:
            return str(key)

    def _get_var(self, getter, args, kw_args):
        if self._path:
            if "name" in kw_args:
                name = f'{self._path}.{kw_args["name"]}'
            else:
                name = self._path
            kw_args["name"] = name
        return getter(*args, **kw_args)

    # Methods for browsing:

    def __getattr__(self, item):
        path = self._resolve_path(item, separator=".")
        return Struct(self._vs, path)

    def __getitem__(self, item):
        if isinstance(item, int) and item < 0:
            item += len(self)
        path = self._resolve_path(f"[{item}]")
        return Struct(self._vs, path)

    def up(self, level=None):
        """Go up one level.

        Args:
            level (str, optional): Assert that the name of the level to go up is equal
                to `level`. Does not perform this assertion if `level` is not given.

        Returns:
            :class:`.spec.Struct`: Struct.
        """
        parts = self._path.split(".")
        if level is not None and parts[-1] != level:
            raise AssertionError(
                f'Cannot go up level "{level}" because the current path is "{self._path}".'
            )
        return Struct(self._vs, ".".join(parts[:-1]))

    def __call__(self):
        return self._vs[self._path]

    def all(self):
        """Get a regex that matches everything in the current path.

        Returns:
            str: Regex.
        """
        return self._resolve_path("*", separator=".")

    # Methods for variable checking and variable manipulation:

    def __bool__(self):
        return self._path in self._vs

    def assign(self, value):
        """Assign a value.

        Args:
            value (tensor): Value to assign.
        """
        self._vs.assign(self._path, value)

    def delete(self):
        """Delete the variable."""
        self._vs.delete(self._path)

    # Methods for container-like behaviour:

    def __len__(self):
        i = 0
        while any(
            name.startswith(self._resolve_path(f"[{i}]")) for name in self._vs.names
        ):
            i += 1
        return i

    def __next__(self):
        return self[len(self)]

    def __iter__(self):
        state = {"counter": -1}

        def get_next():
            state["counter"] += 1
            return self[state["counter"]]

        return (get_next() for _ in repeat(True))


class VariableType(metaclass=ABCMeta):
    """A type of a variable. Any arguments are passed to the appropriate method
    of :class:`.vars.Vars` to instantiate the variable."""

    def __init__(self, *args, **kw_args):
        self.args = args
        self.kw_args = kw_args

    def _get_var(self, method, name, init):
        # Copy dict to not modify the original.
        kw_args = dict(self.kw_args)

        # Set initial value.
        if init is not None and "init" in kw_args and kw_args["init"] is not None:
            raise ValueError(
                f'Initial value doubly specified: "{init}" and "{kw_args["init"]}".'
            )
        kw_args["init"] = init

        # Set name.
        if "name" in kw_args and kw_args["name"] is not None:
            raise ValueError(f'Name doubly specified: {name} and {kw_args["name"]}.')
        kw_args["name"] = name

        return method(*self.args, **kw_args)

    @abstractmethod
    def instantiate(self, vs, name, init):  # pragma: no cover
        """Instantiate the variable.

        Args:
            vs (:class:`.vars.Vars`): Variable container to extract the variable
                from.
            name (str): Name of the variable.
            init (object): Initial value.
        """


class Unbounded(VariableType):
    """Type of an unbounded variable."""

    def instantiate(self, vs, name, init):
        return self._get_var(vs.unbounded, name, init)


class Positive(VariableType):
    """Type of a positive variable."""

    def instantiate(self, vs, name, init):
        return self._get_var(vs.positive, name, init)


class Bounded(VariableType):
    """Type of a bounded variable."""

    def instantiate(self, vs, name, init):
        return self._get_var(vs.bounded, name, init)


class LowerTriangular(VariableType):
    """Type of a lower-triangular matrix."""

    def instantiate(self, vs, name, init):
        return self._get_var(vs.lower_triangular, name, init)


class PositiveDefinite(VariableType):
    """Type of a positive-definite matrix."""

    def instantiate(self, vs, name, init):
        return self._get_var(vs.positive_definite, name, init)


class Orthogonal(VariableType):
    """Type of an orthogonal matrix."""

    def instantiate(self, vs, name, init):
        return self._get_var(vs.orthogonal, name, init)


def parametrised(prefix=""):
    """Decorator to specify variables with types.

    Args:
        prefix (str, optional): Prefix to prepend to the name of generated
            variables. Defaults to no prefix.
    """

    prefix, f = _extract_prefix_and_f(prefix, default="")

    def decorator(f_):
        @wraps(f_)
        def wrapped_f(*args, **kw_args):
            signature = inspect.signature(f_)

            # Convert all arguments to keyword arguments, and store them here.
            filled_kwargs = {}

            # Look for variable container.
            values = args + tuple(kw_args.values())
            num_containers = sum([isinstance(x, Provider) for x in values])
            if num_containers == 0:
                raise ValueError("No variable container found.")
            elif num_containers > 1:
                raise ValueError("Multiple variable containers found.")
            else:
                # There is exactly only variable container. Find it.
                vs = [x for x in values if isinstance(x, Provider)][0]

            # Walk through the arguments.
            for name, parameter in signature.parameters.items():
                annotation = parameter.annotation

                # Instantiate uninstantiated variable types.
                if isinstance(annotation, type) and issubclass(
                    annotation, VariableType
                ):
                    annotation = annotation()

                if isinstance(annotation, VariableType):
                    # Parameter is a variable that needs to be extracted.

                    # Check for default value.
                    if parameter.default is not parameter.empty:
                        init = parameter.default
                    else:
                        init = None

                    # Store the instantiated variable.
                    filled_kwargs[name] = annotation.instantiate(
                        vs, prefix + name, init
                    )

                else:
                    # Parameter is a regular parameter. Find it.

                    if len(args) > 0:
                        # Positional arguments left. Must be next one.
                        filled_kwargs[name] = args[0]
                        args = args[1:]
                    else:
                        # No position arguments left. Extract from keywords.
                        try:
                            filled_kwargs[name] = kw_args[name]
                            del kw_args[name]
                        except KeyError:
                            # Also not found in keywords. Resort to default
                            # value. If a default value is not given,
                            # the user did not specify a positional argument.
                            if parameter.default is parameter.empty:
                                raise ValueError(
                                    f'Positional argument "{name}" not given.'
                                )
                            else:
                                filled_kwargs[name] = parameter.default

            # Ensure that everything is parsed.
            if len(args) > 0:
                raise ValueError(f"{len(args)} positional argument(s) not parsed.")
            if len(kw_args) > 0:
                raise ValueError(
                    f"{len(kw_args)} keyword argument(s) not "
                    f'parsed: {", ".join(kw_args.keys())}.'
                )

            return f_(**filled_kwargs)

        return wrapped_f

    # Return the decorated function directly if a function was given.
    return decorator if f is None else decorator(f)
