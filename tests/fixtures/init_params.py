"""The argument names that a class constructor accepts, also through ``**kwargs``."""
import inspect


def init_params(cls) -> set[str]:
    """Return the argument names that the class constructor accepts.

    Look at a parent class only when the child passes ``**kwargs`` on. Example: a split
    harmonizer with ``__init__(self, *args, **kwargs)`` accepts the arguments of its parent.
    """
    names = set()
    for c in cls.__mro__:
        if c is object:
            break
        params = inspect.signature(c.__init__).parameters
        names |= set(params)
        if not any(p.kind is p.VAR_KEYWORD for p in params.values()):
            break
    return names
