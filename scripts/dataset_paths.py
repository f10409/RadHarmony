"""Read dataset paths from ``configs/data_paths.json``.

The JSON file is the one source of NAS paths in this repo. Each entry is keyed
by a registry key (see ``radharmony.registry.list_datasets``). Its paths are
keyed by constructor argument. So you can pass them to the class directly:

    from dataset_paths import paths
    ds = VinDrCXRTrainDataset(**paths("vindr_cxr_train"), output_cls=True)

Entry fields:
    status    "ok", "restricted" (on the NAS, not readable by all users),
              "not_on_nas" or "alias" (a generic class, so use a split key).
    paths     constructor argument -> path.
    variants  variant name -> {argument -> path}, for one class with more
              than one data folder (for example QaTa-COV19 train and test).
              A variant can also give only the source tables, for the claim
              notes (for example the PadChest-GR variant ``tables``).
    env       constructor argument -> environment variable that overrides it.
    use       for "alias" entries: the split keys to use.
    note      short facts that the paths alone do not tell.

The dev repo commits this file. In a public clone, each user makes their own copy of
``configs/data_paths.example.json`` and fills the entries of the datasets that they want to check.
``RADHARMONY_DATA_PATHS`` gives another file. A missing file or key raises ``MissingPathsError``, which
says what to do.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_CONFIGS = Path(__file__).resolve().parents[1] / "configs"
CONFIG_PATH = Path(os.environ.get("RADHARMONY_DATA_PATHS") or _CONFIGS / "data_paths.json")
EXAMPLE_PATH = _CONFIGS / "data_paths.example.json"
#: The key that ``paths(..., missing_ok=True)`` gives when it has no path.
MISSING = "__missing_paths__"


class MissingPathsError(KeyError):
    """The config file, or its entry for a key, is missing."""

    def __str__(self) -> str:
        return str(self.args[0])


def _hint(key: str) -> str:
    what = f"No path for {key!r}" if has_config() else f"No {CONFIG_PATH.name} at {CONFIG_PATH}"
    return (f"{what}. Copy {EXAMPLE_PATH.name} to {CONFIG_PATH.name} and fill the entries of the datasets "
            "that you want to check (or set RADHARMONY_DATA_PATHS to your file).")


def has_config() -> bool:
    """Return True when the config file exists."""
    return CONFIG_PATH.is_file()


def is_complete() -> bool:
    """Return True when the file says ``"complete": true``: it has an entry for every registry key.

    The dev repo commits a complete file. A user file with only some entries is not complete, so the
    tests that check every key skip.
    """
    if not has_config():
        return False
    with open(CONFIG_PATH) as f:
        return json.load(f).get("complete") is True


@lru_cache(maxsize=None)
def load(config_path: str | Path | None = None) -> dict:
    """Return the ``datasets`` mapping of the config file (default: ``CONFIG_PATH``)."""
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if not path.is_file():
        raise MissingPathsError(f"No paths file at {path}. " + _hint("<any>").split(". ", 1)[1])
    with open(path) as f:
        return json.load(f)["datasets"]


def entry(key: str) -> dict:
    """Return the raw config entry for a registry key."""
    datasets = load()
    if key not in datasets:
        raise MissingPathsError(_hint(key))
    return datasets[key]


def paths(key: str, variant: str | None = None,
          only: tuple[str, ...] | None = None, missing_ok: bool = False) -> dict[str, str]:
    """Return the path arguments for a registry key.

    Args:
        key: registry key, for example ``"vindr_cxr_train"``.
        variant: variant name, for entries that have ``variants``.
        only: keep only these argument names.
        missing_ok: with no file or no entry, return ``{MISSING: <hint>}`` in place of an error.

    Returns:
        ``{argument: path}``, with the environment overrides applied.

    Raises:
        MissingPathsError: the file or the key is missing (a ``KeyError``).
        KeyError: the variant or an argument of ``only`` is not in the entry.
        ValueError: the entry has no paths (``not_on_nas`` or ``alias``).
    """
    try:
        e = entry(key)
    except MissingPathsError as err:
        if missing_ok:
            return {MISSING: str(err)}
        raise
    if e["status"] in ("not_on_nas", "alias"):
        hint = f" Use one of: {', '.join(e['use'])}." if e.get("use") else ""
        raise ValueError(f"{key!r} has status {e['status']!r}.{hint} {e.get('note', '')}".rstrip())
    if variant is None:
        if "paths" not in e:
            raise KeyError(f"{key!r} needs a variant: {', '.join(e['variants'])}")
        out = dict(e["paths"])
    else:
        if variant not in e.get("variants", {}):
            raise KeyError(f"{key!r} has no variant {variant!r}")
        out = dict(e["variants"][variant])
    for arg, var in e.get("env", {}).items():
        if os.environ.get(var):
            out[arg] = os.environ[var]
    if only is not None:
        unknown = set(only) - set(out)
        if unknown:
            raise KeyError(f"{key!r} has no path for {sorted(unknown)}")
        out = {a: out[a] for a in only}
    return out


def path_state(path: str) -> str:
    """Return "ok", "restricted" (a parent folder blocks you) or "missing"."""
    if os.path.exists(path):
        need = os.R_OK | os.X_OK if os.path.isdir(path) else os.R_OK
        return "ok" if os.access(path, need) else "restricted"
    parent = os.path.dirname(path.rstrip("/"))
    while parent and not os.path.exists(parent):
        parent = os.path.dirname(parent)
    return "restricted" if parent and not os.access(parent, os.R_OK | os.X_OK) else "missing"
