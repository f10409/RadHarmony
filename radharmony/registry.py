"""Dataset registry for radharmony.

Allows datasets to be looked up by name and enables third-party datasets
to register themselves without modifying the core package.

Example — built-in usage::

    from radharmony.registry import resolve_dataset, list_datasets

    print(list_datasets())          # ['ct_rate', 'mimic_cxr', 'siim_acr_ptx']
    DatasetCls = resolve_dataset("mimic_cxr")
    dataset = DatasetCls(base_image_dir="/data/mimic", ...)

Example — registering a custom dataset::

    from radharmony.registry import register_dataset
    from radharmony.dataset import BaseRadiologicalDataset

    @register_dataset("my_dataset")
    class MyDataset(BaseRadiologicalDataset):
        _HARMONIZER_CLS = MyDatasetHarmonizer  # for harmonizer_path= / load_from_saved
        ...
"""

_REGISTRY: dict[str, type] = {}


def register_dataset(name: str):
    """Class decorator that registers a dataset under *name*.

    Args:
        name: Registry key used to look up this dataset class.

    Raises:
        ValueError: If *name* is already registered.
    """
    def decorator(cls):
        if name in _REGISTRY:
            raise ValueError(
                f"A dataset named '{name}' is already registered "
                f"({_REGISTRY[name].__qualname__}). Use a different name."
            )
        _REGISTRY[name] = cls
        return cls
    return decorator


def resolve_dataset(name: str) -> type:
    """Return the dataset class registered under *name*.

    Args:
        name: Registry key to look up.

    Returns:
        The registered dataset class.

    Raises:
        KeyError: If *name* is not registered.
    """
    if name not in _REGISTRY:
        available = list_datasets()
        raise KeyError(
            f"No dataset registered as '{name}'. "
            f"Available datasets: {available}"
        )
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    """Return a sorted list of all registered dataset names."""
    return sorted(_REGISTRY.keys())
