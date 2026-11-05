"""Tests for ``pyproject.toml``: an install gives every package that the code imports.

The tests read the imports from the source files and compare them with the
dependency lists. They install nothing.

Run with::

    pytest tests/test_packaging.py -v
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # Python 3.11 or newer

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())
CORE = PYPROJECT["project"]["dependencies"]
EXTRAS = PYPROJECT["project"]["optional-dependencies"]

# Import name -> distribution name, for the packages that the dataset code uses.
CORE_MODULES = {
    "PIL": "pillow", "SimpleITK": "simpleitk", "cv2": "opencv-python-headless",
    "itk": "itk", "matplotlib": "matplotlib", "monai": "monai", "numpy": "numpy",
    "pandas": "pandas", "pycocotools": "pycocotools", "pydicom": "pydicom", "torch": "torch",
    "tqdm": "tqdm",
}
# matplotlib and monai install Pillow. The core dependency monai[itk] installs itk.
TRANSITIVE = {"pillow", "itk"}

# Backbone module -> the extra that the Install page (docs/wiki/get-started/install.md) names for it.
BACKBONE_EXTRA = {
    "ark_plus": "ark_plus", "biomed_clip": "biomed", "chexfound": "chexfound",
    "eva_x": "eva_x", "medical_mae": "medical_mae", "medimageinsights": "medimageinsights",
}


def _dist_names(requirements: list[str]) -> set[str]:
    """``"monai[itk]"`` -> ``"monai"``, ``"torch>=2"`` -> ``"torch"``."""
    return {re.split(r"[\[<>=!~; @]", r.strip(), maxsplit=1)[0].lower().replace("_", "-")
            for r in requirements}


def _imports(path: Path) -> set[str]:
    """Top-level names of the absolute imports in a file."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names - set(sys.stdlib_module_names) - {"radharmony", "radharmony_internal"}


def _dataset_files() -> list[Path]:
    pkg = ROOT / "radharmony"
    return (list(pkg.glob("*.py")) + list((pkg / "dataset").rglob("*.py"))
            + list((pkg / "harmonizer").rglob("*.py")) + list((pkg / "preprocessor").rglob("*.py")))


def test_dataset_code_needs_only_core_dependencies():
    """V-A-12, V-H2-5: RANZCR masks import cv2, so a plain install must give it."""
    core = _dist_names(CORE) | TRANSITIVE
    missing = {}
    for f in _dataset_files():
        for mod in _imports(f):
            dist = CORE_MODULES.get(mod)
            assert dist, f"{f.relative_to(ROOT)} imports {mod!r}: add it to CORE_MODULES"
            if dist not in core:
                missing.setdefault(dist, []).append(str(f.relative_to(ROOT)))
    assert not missing, missing


@pytest.mark.parametrize("module, extra", sorted(BACKBONE_EXTRA.items()))
def test_backbone_extra_lists_torchvision(module, extra):
    """V-A-11, A-7: a backbone that imports torchvision lists it in its extra."""
    if "torchvision" not in _imports(ROOT / "radharmony/evaluator/backbones" / f"{module}.py"):
        pytest.skip(f"{module} does not import torchvision")
    assert "torchvision" in _dist_names(EXTRAS[extra]) | _dist_names(CORE)


def test_install_page_says_radeval_needs_python_311():
    """A-11: the radeval extra installs nothing below Python 3.11. The Install page is the home of the extras (6c)."""
    assert any("python_version >= '3.11'" in r for r in EXTRAS["radeval"])
    row = next(line for line in (ROOT / "docs/wiki/get-started/install.md").read_text().splitlines()
               if line.startswith("| `radeval`"))
    assert "3.11" in row, row
