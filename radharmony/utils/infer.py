"""Path inference utility for dataset CSV auto-discovery."""

import os


def infer_path(
    base_dir: str,
    *candidates: str,
    user_path: str = "",
    required: bool = False,
    arg: str = "",
) -> str:
    """Find a source file in exact places near ``base_dir`` (D-35).

    Search order:

    1. ``user_path``, if given. A ``user_path`` that does not exist raises
       ``FileNotFoundError``.
    2. ``base_dir/<candidate>``, for each candidate in order.
    3. ``parent/<candidate>``, where ``parent`` is the folder above ``base_dir``.

    It does not walk folders, because a walk can take a file of another
    dataset with the same name. A candidate can have a sub folder, for example
    ``annotations/image_labels_train.csv``.

    Args:
        base_dir: Root image directory (the anchor of the search).
        *candidates: File names or relative paths to look for, in order.
        user_path: Explicit path from the caller. It must exist.
        required: Raise ``FileNotFoundError`` when no candidate is found.
        arg: The constructor argument that gives this file. The error names it.

    Returns:
        The path that was found, or ``""``.

    Raises:
        FileNotFoundError: ``user_path`` does not exist, or ``required`` is
            True and no candidate is found.
    """
    if user_path:
        if not os.path.exists(user_path):
            raise FileNotFoundError(f"{arg or 'The given path'} does not exist: {user_path}")
        return user_path
    if not base_dir:
        if required:
            raise FileNotFoundError(f"{' or '.join(candidates)} needs base_image_dir or {arg or 'a path'}.")
        return ""

    parent = os.path.normpath(os.path.join(base_dir.rstrip("/\\"), ".."))
    for folder in (base_dir, parent):
        for c in candidates:
            path = os.path.normpath(os.path.join(folder, c))  # "a/b/../c" becomes "a/c"
            if os.path.exists(path):
                return path
    if required:
        hint = f" Pass {arg}=<path>." if arg else ""
        raise FileNotFoundError(f"{' or '.join(candidates)} is not in {base_dir} or {parent}.{hint}")
    return ""
