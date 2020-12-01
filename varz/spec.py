import inspect
from abc import ABCMeta, abstractmethod
from functools import wraps
from inspect import isfunction

from .vars import Provider

__all__ = [
    "sequential",
    "Unbounded",
    "Positive",
    "Bounded",
    "LowerTriangular",
    "PositiveDefinite",
    "Orthogonal",
    "parametrised",
]


class Sequential(Provider):
    """A variable provider that wraps a :class:`.vars.Vars` object and
    automatically names unnamed variables sequentially (0, 1, 2, ...) with a
    possible prefix.

    Args:
        vs (:class:`.vs.Vars`): Variable container to wrap.
        prefix (str): Prefix for names in the sequence.
    """

    def __init__(self, vs, prefix):
        self.vs = vs
        self.prefix = prefix
        self.count = 0

    def _get_var(self, getter, args, kw_args):
        if "name" not in kw_args:
            kw_args["name"] = f"{self.prefix}{self.count}"
            self.count += 1
        return getter(*args, **kw_args)

    def unbounded(self, *args, **kw_args):
        return self._get_var(self.vs.unbounded, args, kw_args)

    def positive(self, *args, **kw_args):
        return self._get_var(self.vs.positive, args, kw_args)

    def bounded(self, *args, **kw_args):
        return self._get_var(self.vs.bounded, args, kw_args)

    def lower_triangular(self, *args, **kw_args):
        return self._get_var(self.vs.lower_triangular, args, kw_args)

    def positive_definite(self, *args, **kw_args):
        return self._get_var(self.vs.positive_definite, args, kw_args)

    def orthogonal(self, *args, **kw_args):
        return self._get_var(self.vs.orthogonal, args, kw_args)

    def __getitem__(self, name):
        return self.vs[name]


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


def _extract_prefix_and_f(prefix_or_f):
    """Extract the prefix and function.

    Args:
        prefix_or_f (object): Either a prefix or a function.

    Returns:
        tuple: Tuple containing the predict and the function. If a function
            was given, the prefix defaults to an empty string; and if a prefix
            was given, the function defaults to `None`.
    """
    if prefix_or_f is None:
        # Not used as decorator and prefix is left unspecified.
        return "", None
    elif isfunction(prefix_or_f):
        # Used as a decorator.
        return "", prefix_or_f
    else:
        # Not used as decorator and prefix is specified.
        return prefix_or_f, None


def sequential(prefix=None):
    """Decorator that generates variable names for unnamed variables
    sequentially.

    Args:
        prefix (str, optional): Prefix to prepend to the name of generated
            variables. Defaults to no prefix.
    """
    prefix, f = _extract_prefix_and_f(prefix)

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
        return self._get_var(vs.get, name, init)


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


def parametrised(prefix=None):
    """Decorator to specify variables with types.

    Args:
        prefix (str, optional): Prefix to prepend to the name of generated
            variables. Defaults to no prefix.
    """

    prefix, f = _extract_prefix_and_f(prefix)

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
