"""Evaluator registry for radharmony.

Separate from the dataset registry so evaluator concerns don't pollute
`radharmony.registry`.

Example::

    from radharmony.evaluator import resolve_evaluator, list_evaluators

    print(list_evaluators())  # ['finetune', 'knn_probe', 'linear_probe', 'zero_shot']
    EvalCls = resolve_evaluator("linear_probe")
"""

_REGISTRY: dict[str, type] = {}


def register_evaluator(name: str):
    """Class decorator that registers an evaluator under *name*.

    Raises:
        ValueError: If *name* is already registered.
    """
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(
                f"An evaluator named '{name}' is already registered "
                f"({_REGISTRY[name].__qualname__}). Use a different name."
            )
        _REGISTRY[name] = cls
        return cls
    return decorator


def resolve_evaluator(name: str) -> type:
    """Return the evaluator class registered under *name*.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        available = list_evaluators()
        raise KeyError(
            f"No evaluator registered as '{name}'. "
            f"Available evaluators: {available}"
        )
    return _REGISTRY[name]


def list_evaluators() -> list[str]:
    """Return a sorted list of all registered evaluator names."""
    return sorted(_REGISTRY.keys())
