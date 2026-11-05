"""Official MIMIC-CXR splits (``mimic-cxr-2.0.0-split.csv``), shared by the DICOM and JPG classes."""

import os

import pandas as pd

SPLIT_FILE = "mimic-cxr-2.0.0-split.csv"


def find_next_to(name: str, *paths: str) -> str | None:
    """Return ``name`` (or ``name.gz``) in the folder of the first path that has it, else None."""
    for p in paths:
        if not p:
            continue
        for candidate in (name, name + ".gz"):
            hit = os.path.join(os.path.dirname(p), candidate)
            if os.path.isfile(hit):
                return hit
    return None


class _MIMICSplitMixin:
    """Keep the rows of one official split, by ``dicom_id``.

    Put this class before the MIMIC harmonizer in the class list. It removes
    the other rows right after it builds the IDs, so ``self.df`` holds only
    the split (D-24).

    The split is kept as published (D-34). Example: 3 patients of the
    validate split are also in the labeled test set of version 2.1.0.

    Class attributes:
        SPLIT: ``"train"``, ``"validate"`` or ``"test"``, as in the split file.
    """

    SPLIT: str = None

    def _split_file(self, name: str, user_path: str | None, arg: str) -> str:
        """Return ``user_path``, or find ``name`` next to the source CSVs. Raise if none."""
        path = user_path or find_next_to(name, self.csv_path, self.label_csv_path)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(
                f"{type(self).__name__} needs {name}, but it is not next to csv_path or "
                f"label_csv_path. Pass {arg}=..."
            )
        return path

    def _build_study_id(self) -> None:
        super()._build_study_id()
        split = pd.read_csv(self._split_file(SPLIT_FILE, self.split_csv_path, "split_csv_path"))
        keep = set(split.loc[split["split"] == self.SPLIT, "dicom_id"])
        self.df = self.df[self.df["dicom_id"].isin(keep)].reset_index(drop=True)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if getattr(self, "split_csv_path", None) is not None:
            snap["split_csv_path"] = self.split_csv_path
        return snap
