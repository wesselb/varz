import inspect
from abc import ABCMeta, abstractmethod
from functools import wraps

from .vars import Vars


class VarsProxy:
    """A proxy for a :class:`.vars.Vars` object.

    Args:
        vs (:class:`.vs.Vars`): Variable container to wrap.
    """

    def __init__(self, vs):
        self.vs = vs

    def __getattr__(self, item):
        return getattr(self.vs, item)


def is_vars(x):
    return isinstance(x, (Vars, VarsProxy))


class Sequential(VarsProxy):
    """A proxy for a :class:`.vars.Vars` object that automatically names
    unnamed variables sequentially.

    Args:
        vs (:class:`.vs.Vars`): Variable container to wrap.
        prefix (str): Prefix for names in the sequence.
    """

    def __init__(self, vs, prefix):
        VarsProxy.__init__(self, vs)
        self.prefix = prefix
        self.count = 0

    def _get_var(self, method, args, kw_args):
        if 'name' not in kw_args:
            kw_args['name'] = f'{self.prefix}{self.count}'
            self.count += 1
        return getattr(self.vs, method)(*args, **kw_args)

    def get(self, *args, **kw_args):
        return self._get_var('get', args, kw_args)

    def positive(self, *args, **kw_args):
        return self._get_var('positive', args, kw_args)

    def pos(self, *args, **kw_args):
        return self._get_vars('pos', *args, **kw_args)

    def bounded(self, *args, **kw_args):
        return self._get_var('bounded', args, kw_args)

    def bnd(self, *args, **kw_args):
        return self._get_var('bnd', *args, **kw_args)


def _to_sequential(x, prefix):
    """Convert an variable container to sequential, if it is one.

    Args:
        x (object): Object to convert.
        prefix (str): Prefix for names in the sequence.

    Returns:
        object: `x` converted to sequential if it is a variable container,
            otherwise just `x`.
    """
    return Sequential(x, prefix) if is_vars(x) else x


def _extract_prefix_and_f(prefix_or_f):
    if prefix_or_f is None:
        # Not used as decorator and prefix is left unspecified.
        prefix = ''
        f = None
    elif isinstance(prefix_or_f, str):
        # Not used as decorator and prefix is specified.
        prefix = prefix_or_f
        f = None
    else:
        # Used as a decorator.
        prefix = ''
        f = prefix_or_f

    return prefix, f


def sequential(prefix_or_f=None):
    """Decorator that generates variable names for unnamed variables
    sequentially."""
    prefix, f = _extract_prefix_and_f(prefix_or_f)

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
    def __init__(self, *args, **kw_args):
        self.args = args
        self.kw_args = kw_args

    def _get_var(self, method, name, init):
        # Copy dict to not modify the original.
        kw_args = dict(self.kw_args)

        # Set initial value.
        if (
                init is not None and
                'init' in kw_args and
                kw_args['init'] is not None
        ):
            raise ValueError(f'Initial value doubly specified: '
                             f'"{init}" and "{kw_args["init"]}".')
        kw_args['init'] = init

        # Set name.
        if name in kw_args and kw_args['name'] is not None:
            raise ValueError(f'Name doubly specified: '
                             f'{name} and {kw_args["name"]}.')
        kw_args['name'] = name

        return method(*self.args, **kw_args)

    @abstractmethod
    def instantiate(self, vs, name, init):  # pragma: no cover
        pass


class Unbounded(VariableType):
    def instantiate(self, vs, name, init):
        return self._get_var(vs.get, name, init)


class Positive(VariableType):
    def instantiate(self, vs, name, init):
        return self._get_var(vs.positive, name, init)


class Bounded(VariableType):
    def instantiate(self, vs, name, init):
        return self._get_var(vs.bounded, name, init)


def parametrised(prefix_or_f=None):
    """Decorator to specify variables with types."""

    prefix, f = _extract_prefix_and_f(prefix_or_f)

    def decorator(f_):
        @wraps(f_)
        def wrapped_f(*args, **kw_args):
            signature = inspect.signature(f_)

            # Convert all arguments to keyword arguments, and store them here.
            filled_kwargs = {}

            # Look for variable container.
            values = args + tuple(kw_args.values())
            num_containers = sum([is_vars(x) for x in values])
            if num_containers == 0:
                raise ValueError('No variable container found.')
            elif num_containers > 1:
                raise ValueError('Multiple variable containers found.')
            else:
                # There is exactly only variable container. Find it.
                vs = [x for x in values if is_vars(x)][0]

            # Walk through the arguments.
            for name, parameter in signature.parameters.items():
                annotation = parameter.annotation

                # Instantiate uninstantiated variable types.
                if (isinstance(annotation, type) and
                        issubclass(annotation, VariableType)):
                    annotation = annotation()

                # Check whether the parameter is a variable that needs to be
                # extracted.
                if isinstance(annotation, VariableType):
                    # Check for default value.
                    if parameter.default is not parameter.empty:
                        init = parameter.default
                    else:
                        init = None

                    # Store the instantiated variable.
                    filled_kwargs[name] = \
                        annotation.instantiate(vs, prefix + name, init)
                else:
                    if len(args) > 0:
                        # Positional arguments left. Must be next one.
                        filled_kwargs[name] = args[0]
                        args = args[1:]
                    else:
                        # No position arguments left. Extract from keywords.
                        filled_kwargs[name] = kw_args[name]
                        del kw_args[name]

            # Ensure that everything is parsed.
            if len(args) > 0:
                raise ValueError(f'{len(args)} positional argument(s) not '
                                 f'parsed.')
            if len(kw_args) > 0:
                raise ValueError(f'{len(kw_args)} keyword argument(s) not '
                                 f'parsed: {", ".join(kw_args.keys())}.')

            return f_(**filled_kwargs)

        return wrapped_f

    # Return the decorated function directly if a function was given.
    return decorator if f is None else decorator(f)
