"""Print one count for a claim note: from ``expected_stats``, or from a source file.

Table counts (tier=ci). ``tests/test_dataset_stats.py`` checks ``expected_stats`` against
``harmonize()`` on the real data. So a claim that prints a value from it ties the docs to
that test::

    python scripts/evalkit/dataset_stat.py chexpert chexpert_train.images
    # 138,358

The first argument is the file name in ``configs/datasets/`` without ``.json``. The second
is a dotted path in ``expected_stats``. A whole number prints with commas. A sum of
several paths prints when they are joined with ``+``, for example
``chexpert_train.images+chexpert_valid.images``.

Source counts (tier=nas). ``--source`` reads a source file on the NAS, with the path from
``configs/data_paths.json``::

    python scripts/evalkit/dataset_stat.py --source chexpert_train csv_path rows
    # 223,414
    python scripts/evalkit/dataset_stat.py --source chexpert_train csv_path unique:Sex

The arguments are:

- the registry key, or ``<key>:<variant>`` for a variant of the entry.
- the path argument, or ``<argument>/<relative file>`` for a file in a folder argument.
- the count: ``rows`` (the rows of a CSV or Excel file, or the items of a JSON file) or
  ``unique:<column>``. A filter ``@<column>=<value>`` at the end keeps only the matching rows,
  for example ``unique:ImageID@split=train``.
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root first: test this clone, not an installed copy
REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "configs" / "datasets"
#: The exit code when the data paths file has no path for the key. claims.py skips such a claim.
NO_PATH = 3
sys.path.insert(0, str(REPO / "scripts"))
import dataset_paths  # noqa: E402  (module calls, so a test can replace paths)


def stat(config: str, path: str):
    """Return the value at ``path`` in the ``expected_stats`` of ``config``.

    Args:
        config: File name in ``configs/datasets/``, without ``.json``.
        path: Dotted path, or several dotted paths joined with ``+`` (their sum).

    Returns:
        The value. A sum is an int.
    """
    stats = json.loads((CONFIGS / f"{config}.json").read_text(encoding="utf-8"))["expected_stats"]
    values = []
    for part in path.split("+"):
        node = stats
        for key in part.split("."):
            node = node[key]
        values.append(node)
    return sum(values) if len(values) > 1 else values[0]


def _read_table(path: str):
    """Read a CSV (also .gz), Excel or JSON file. A JSON list or object stays as it is."""
    import pandas as pd

    if path.endswith((".json", ".json.gz")):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path, low_memory=False)


def source_count(key: str, arg: str, count: str) -> int:
    """Count the rows, or the distinct values of a column, of a source file.

    Args:
        key: Registry key in ``configs/data_paths.json``, or ``<key>:<variant>``.
        arg: Path argument, or ``<argument>/<relative file>`` for a file in a folder argument.
        count: ``rows`` or ``unique:<column>``, with an optional ``@<column>=<value>`` filter.

    Returns:
        The count.
    """
    key, _, variant = key.partition(":")
    name, _, rel = arg.partition("/")
    path = dataset_paths.paths(key, variant or None)[name]
    path = str(Path(path) / rel) if rel else path
    table = _read_table(path)
    count, _, where = count.partition("@")
    if where:
        if not hasattr(table, "columns"):
            raise ValueError("a filter needs a table, not a JSON file")
        col, _, value = where.partition("=")
        table = table[table[col].astype(str) == value]
        if table.empty:  # a wrong value must fail, not give 0
            raise ValueError(f"no row has {col} = {value!r}")
    if count == "rows":
        return len(table)
    if count.startswith("unique:") and hasattr(table, "columns"):
        return int(table[count[len("unique:"):]].nunique())
    raise ValueError(f"unknown count {count!r}: use rows or unique:<column> (unique needs a table)")


def fmt(value) -> str:
    """Format a value as the docs write it: a whole number with commas."""
    if isinstance(value, int) and not isinstance(value, bool):
        return f"{value:,}"
    return json.dumps(value) if isinstance(value, (list, dict)) else str(value)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--source"] and len(argv) == 4:
        try:
            print(fmt(source_count(*argv[1:])))
        except dataset_paths.MissingPathsError as e:  # no data paths file, or no entry: claims.py skips the claim
            print(e, file=sys.stderr)
            return NO_PATH
        return 0
    if len(argv) != 2 or argv[0].startswith("-"):
        print(__doc__, file=sys.stderr)
        return 2
    print(fmt(stat(*argv)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
