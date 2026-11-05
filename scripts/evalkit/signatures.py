"""Find a radharmony name from a doc import, and the arguments that it takes.

``run_blocks.py`` uses it to check the calls in the doc code blocks. ``wiki_audit.py`` uses
it to check the argument tables. Neither runs the doc code. Both import only the package.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from dataclasses import dataclass

from blocks import PY_LANGS, python_source

_MOD_CACHE: dict[str, object] = {}


def _import(mod: str):
    if mod not in _MOD_CACHE:
        _MOD_CACHE[mod] = importlib.import_module(mod)
    return _MOD_CACHE[mod]


def resolve_symbol(module: str, name: str):
    """Return the object of ``from <module> import <name>``, or None.

    Only radharmony modules are read. Any other module gives None.
    """
    if not module.startswith("radharmony"):
        return None
    try:
        return getattr(_import(module), name)
    except Exception:  # noqa: BLE001  (any import error means "does not resolve")
        pass
    # A submodule, for example ``from radharmony.evaluator.backbones import make_x``.
    try:
        return _import(f"{module}.{name}")
    except Exception:  # noqa: BLE001
        return None


def page_imports(blocks) -> tuple[dict, list[tuple[int, str, str]]]:
    """Read the radharmony imports of all the Python blocks of a page.

    An import anywhere on the page counts for every block of the page. A block that does
    not compile is left out.

    Returns:
        ``({local name: object}, [(Markdown line, module, name) of each import that does not resolve]``.
    """
    names, bad = {}, []
    for b in blocks:
        if b.lang not in PY_LANGS:
            continue
        code, lines = python_source(b)
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("radharmony")):
                continue
            for alias in node.names:
                obj = resolve_symbol(node.module, alias.name)
                if obj is None:
                    bad.append((lines[node.lineno - 1], node.module, alias.name))
                else:
                    names[alias.asname or alias.name] = obj
    return names, bad


@dataclass
class SigModel:
    """The keyword arguments of a function or a class.

    Attributes:
        params: name -> ``inspect.Parameter``. For a class, the subclass wins over a base class.
        closed: False when the last ``__init__`` still has ``**kwargs``. Then any name can be valid.
    """

    params: dict[str, inspect.Parameter]
    closed: bool


def signature_model(obj) -> SigModel | None:
    """Return the arguments of a class or a function, or None when there is no signature."""
    if inspect.isclass(obj):
        return _class_sig(obj)
    if callable(obj):
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            return None
        params, var_kw = _collect(sig)
        return SigModel(params=params, closed=not var_kw)
    return None


def _collect(sig: inspect.Signature):
    params: dict[str, inspect.Parameter] = {}
    var_kw = False
    for name, p in sig.parameters.items():
        if name == "self" or p.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            var_kw = True
            continue
        params[name] = p
    return params, var_kw


def _class_sig(cls) -> SigModel | None:
    """Return the arguments of a class.

    The arguments of a base class count only when the subclass passes ``**kwargs`` on,
    as the evaluators do with ``**base_kwargs``. A dataset class names all its arguments,
    so its base adds nothing.
    """
    inits = [k for k in cls.__mro__ if k is not object and "__init__" in vars(k)]
    if not inits:
        try:
            sig = inspect.signature(cls)
        except (TypeError, ValueError):
            return None
        params, var_kw = _collect(sig)
        return SigModel(params=params, closed=not var_kw)
    merged: dict[str, inspect.Parameter] = {}
    forwards = True  # go on to the next base only while the last __init__ had **kwargs
    closed = True
    for klass in inits:  # subclass first
        if not forwards:
            break
        try:
            sig = inspect.signature(klass.__init__)
        except (TypeError, ValueError):
            continue
        params, var_kw = _collect(sig)
        for name, p in params.items():
            merged.setdefault(name, p)
        forwards = var_kw
        closed = not var_kw
    return SigModel(params=merged, closed=closed)
