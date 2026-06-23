"""Path inference utility for dataset CSV auto-discovery."""

import glob as _glob
import os


def infer_path(base_dir: str, *candidates: str, user_path: str = "") -> str:
    """Resolve a file path near ``base_dir`` with progressive fallback.

    Resolution order:

    1. ``user_path`` — if given and exists on disk, return it immediately.
    2. Exact ``base_dir/candidate`` for each candidate.
    3. Exact ``parent/candidate``, then recursive glob inside each *sibling*
       of ``base_dir`` (children of parent, excluding ``base_dir`` itself).
    4. Recursive glob inside each *uncle* of ``base_dir`` (children of
       grandparent, excluding parent) — only reached if step 3 finds nothing.

    Steps 3–4 never descend into ``base_dir``, avoiding the image tree.

    Args:
        base_dir: Root image directory (used as the anchor for the search).
        *candidates: File names or relative paths to look for (in order).
        user_path: Explicit path supplied by the caller; takes priority if it
            exists on disk.

    Returns:
        The resolved path string, or ``""`` if nothing was found.
    """
    if not base_dir:
        return user_path if (user_path and os.path.exists(user_path)) else ""

    # 1. User-provided path exists — use it immediately
    if user_path and os.path.exists(user_path):
        return user_path

    # 2. Exact match directly under base_dir
    for c in candidates:
        p = os.path.join(base_dir, c)
        if os.path.exists(p):
            return p

    def _glob_dirs(dirs, exclude):
        for d in dirs:
            if not d.is_dir():
                continue
            try:
                if os.path.samefile(d.path, exclude):
                    continue
            except FileNotFoundError:
                continue
            for c in candidates:
                matches = _glob.glob(
                    os.path.join(d.path, "**", os.path.basename(c)), recursive=True
                )
                if matches:
                    return matches[0]
        return ""

    parent = os.path.normpath(os.path.join(base_dir.rstrip("/\\"), ".."))
    grandparent = os.path.normpath(os.path.join(parent, ".."))

    # 3. Exact parent/candidate, then sibling dirs (recursive glob)
    if os.path.isdir(parent):
        for c in candidates:
            p = os.path.join(parent, c)
            if os.path.exists(p):
                return p
        hit = _glob_dirs(os.scandir(parent), exclude=base_dir)
        if hit:
            return hit

    # 4. Uncle dirs (recursive glob) — only if step 3 found nothing
    if os.path.isdir(grandparent):
        hit = _glob_dirs(os.scandir(grandparent), exclude=parent)
        if hit:
            return hit

    return ""
