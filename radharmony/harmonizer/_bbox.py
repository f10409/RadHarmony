"""The shared box steps (D-07): clip a box to the image, drop a box with no area, name a box label."""

import json
import math
import warnings


def box_label(name) -> str:
    """Return the snake case name of a box label, as for the label columns (D-05, D-07, Q-17).

    Example: ``"Pleural effusion"`` gives ``"pleural_effusion"``. A ``/`` stays, as in the
    label column ``nodule/mass``.
    """
    return "_".join(str(name).lower().split())


def clip_box(box: list) -> list | None:
    """Return ``box`` clipped to [0, 1], or ``None`` when no area is left (D-07).

    No area means that a min is not less than its max after the clip. Examples: a box with
    no width, an inverted box (min > max), or a box past the image edge.

    Args:
        box: fractions ``[dim0_min, dim0_max, dim1_min, dim1_max, ...]``.

    Returns:
        The clipped box. A box with a NaN or inf value is returned as the same object,
        so that the transform raises on it (D-07).
    """
    values = [float(v) for v in box]
    if not all(math.isfinite(v) for v in values):
        return box
    values = [min(1.0, max(0.0, v)) for v in values]
    if any(lo >= hi for lo, hi in zip(values[0::2], values[1::2], strict=True)):
        return None
    return values


def _read(cell):
    """Return ``(value, was_text)``. A JSON text cell gives its parsed value."""
    if isinstance(cell, str):
        return json.loads(cell), True
    return cell, False


def warn_dropped(name: str, n: int, labels: bool = True) -> None:
    """Warn with the number ``n`` of dropped boxes (D-07). Nothing when ``n`` is 0."""
    if n:
        with_labels = ", with their labels" if labels else ""
        warnings.warn(f"{name}: dropped {n} box{'es' if n > 1 else ''} with no area inside the image"
                      f"{with_labels} (D-07).", UserWarning, stacklevel=3)


def clip_boxes(df, name: str, col: str = "bbox", per_box: tuple = ("bbox_labels", "bbox_findings"),
               count_col: str = "n_boxes") -> int:
    """Clip each box of ``df[col]`` in place, and drop the boxes with no area (D-07).

    A cell keeps its form (a list or JSON text). A cell with no boxes, and a box that is
    already inside the image, stay as they are.

    Args:
        df: harmonized table. A table with no ``col`` column is not changed.
        name: dataset name for the warning.
        col: the box column.
        per_box: the columns with one entry for each box. The entries of a dropped box go too.
        count_col: the column with the number of boxes. It gets the new number.

    Returns:
        The number of dropped boxes. If it is not 0, a ``UserWarning`` gives it.
    """
    if col not in df.columns:
        return 0
    dropped = 0
    per_box = [c for c in per_box if c in df.columns]
    for i, cell in df[col].items():
        if not isinstance(cell, (str, list, tuple)):
            continue  # no boxes
        boxes, text = _read(cell)
        if not boxes:
            continue
        flat = not isinstance(boxes[0], (list, tuple))
        old = [boxes] if flat else boxes
        kept = [clip_box(b) for b in old]
        keep = [b is not None for b in kept]
        if all(keep) and all(k is b or k == [float(v) for v in b] for k, b in zip(kept, old, strict=True)):
            continue  # every box is inside the image
        dropped += keep.count(False)
        new = [b for b in kept if b is not None]
        value = (new[0] if new else []) if flat else new
        df.at[i, col] = json.dumps(value) if text else value
        if all(keep):
            continue
        if count_col in df.columns:
            df.at[i, count_col] = len(new)
        for c in per_box:
            entries, c_text = _read(df.at[i, c]) if isinstance(df.at[i, c], (str, list)) else (None, False)
            if isinstance(entries, list) and len(entries) == len(keep):
                entries = [e for e, k in zip(entries, keep, strict=True) if k]
                df.at[i, c] = json.dumps(entries) if c_text else entries
    warn_dropped(name, dropped, labels=bool(per_box))
    return dropped
