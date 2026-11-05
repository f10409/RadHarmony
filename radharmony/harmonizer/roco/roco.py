"""Harmonizer for the ROCO (Radiology Objects in COntext) dataset.

ROCO is a large multimodal captioning set from PubMed Central. An entry has a radiology or
non-radiology image, its figure caption, and UMLS keywords, CUIs and semantic types.
Source: https://github.com/razorx89/roco-dataset

Layout after ``scripts/fetch.py``::

    <base_dir>/data/{train,validation,test}/{radiology,non-radiology}/
      captions.txt   # ROCO_XXXXX<TAB><caption>
      keywords.txt   # ROCO_XXXXX<TAB><TAB><kw1><TAB><kw2>...  (first field blank)
      cuis.txt
      semtypes.txt
      images/ROCO_XXXXX.jpg

Columns (VQA schema): patient_id, study_id, question_id, image_path, question, answer.
``question`` is always empty, because this is a captioning set. ``answer`` is the caption.
Extra columns: ``keywords`` (tab-joined), ``split`` and ``subset`` ("radiology" or
"non-radiology").

Details: ``docs/wiki/reference/datasets/vqa-text/roco.md``.
"""

import os
import warnings

import pandas as pd

from ..base_vqa import BaseVQAHarmonizer


_SPLITS   = ("train", "validation", "test")
_SUBSETS  = ("radiology", "non-radiology")


def _parse_tsv(path: str, min_fields: int = 2) -> list[tuple]:
    """Read a ROCO tab-separated file and return the list of split lines (the id is first)."""
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= min_fields:
                rows.append(parts)
    return rows


class ROCOHarmonizer(BaseVQAHarmonizer):
    """Harmonize the ROCO dataset.

    Args:
        base_dir: Root of the roco-dataset clone or download, the folder with ``data/``.
        splits: The splits to use. The default is all three: ``["train", "validation", "test"]``.
        radiology_only: If ``True`` (default), use only the ``radiology/`` subset. Use ``False``
            to add the ``non-radiology/`` images.
        image_subdir: Name of the image folder in each split and subset folder (default
            ``"images"``). It must match the ``--subdir`` argument of ``scripts/fetch.py``.
    """

    EXTRA_OUTPUT_COLS = ["keywords", "split", "subset"]

    def __init__(
        self,
        base_dir: str = None,
        splits: list[str] | None = None,
        radiology_only: bool = True,
        image_subdir: str = "images",
    ):
        super().__init__(base_image_dir=base_dir)
        self.base_dir = os.path.expanduser(base_dir) if base_dir else None
        self.splits = list(splits) if splits else list(_SPLITS)
        self.radiology_only = radiology_only
        self.image_subdir = image_subdir

        unknown = set(self.splits) - set(_SPLITS)
        if unknown:
            raise ValueError(f"Unknown splits: {unknown}. Must be in {_SPLITS}")

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        # The constructor sets base_image_dir from base_dir. It is not an argument (V-H2-22).
        snap.pop("base_image_dir", None)
        if self.base_dir is not None:
            snap["base_dir"] = self.base_dir
        if self.splits != list(_SPLITS):
            snap["splits"] = self.splits
        if not self.radiology_only:
            snap["radiology_only"] = False
        if self.image_subdir != "images":
            snap["image_subdir"] = self.image_subdir
        return snap

    def filter_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep the rows of ``splits``, and only the ``radiology`` subset if ``radiology_only`` (D-02).

        ``_read_source`` reads only these folders. This method does the same on a harmonized table.

        Args:
            df: The source rows or a harmonized table.

        Returns:
            The kept rows, with a new index.

        Raises:
            ValueError: If ``radiology_only`` is set and the table has no ``subset`` column.
        """
        # A table with no split column is taken as it is.
        if set(self.splits) != set(_SPLITS) and "split" in df.columns:
            df = df[df["split"].isin(self.splits)]
        if self.radiology_only:
            if "subset" not in df.columns:
                raise ValueError("ROCO: radiology_only=True needs the subset column, and the table has none.")
            df = df[df["subset"] == "radiology"]
        return df.reset_index(drop=True)

    def _read_source(self) -> pd.DataFrame:
        """Read the caption and keyword files of the chosen splits and subsets, one row per image."""
        if self.base_dir is None:
            raise ValueError("base_dir is required to harmonize from scratch.")

        subsets = ("radiology",) if self.radiology_only else _SUBSETS

        rows = []
        missing = []
        for split in self.splits:
            for subset in subsets:
                folder = os.path.join(self.base_dir, "data", split, subset)
                if not os.path.isdir(folder):
                    missing.append(os.path.join("data", split, subset))
                    continue

                cap_path = os.path.join(folder, "captions.txt")
                kw_path  = os.path.join(folder, "keywords.txt")
                if not os.path.isfile(cap_path):  # V-H2-45: a broken download lost a split with no message
                    missing.append(os.path.join("data", split, subset, "captions.txt"))
                    continue

                # Keywords: id to joined keyword text.
                kw_map: dict[str, str] = {}
                for parts in _parse_tsv(kw_path, min_fields=1):
                    roco_id = parts[0].strip()
                    # The keywords start at index 2. Index 1 is always blank.
                    kws = [p.strip() for p in parts[2:] if p.strip()]
                    kw_map[roco_id] = "\t".join(kws)

                for parts in _parse_tsv(cap_path, min_fields=2):
                    roco_id = parts[0].strip()
                    caption = "\t".join(parts[1:]).strip()
                    if not roco_id or not caption:
                        continue

                    img_rel = os.path.join(
                        "data", split, subset, self.image_subdir,
                        f"{roco_id}.jpg"
                    )

                    rows.append({
                        "patient_id":  roco_id,
                        "study_id":    roco_id,
                        "question_id": roco_id,
                        "image_path":  img_rel,
                        "question":    "",
                        "answer":      caption,
                        "keywords":    kw_map.get(roco_id, ""),
                        "split":       split,
                        "subset":      subset,
                    })

        # D-35: no folder with captions.txt is an error. A missing folder or captions.txt is a warning.
        n_folders = len(self.splits) * len(subsets)
        if len(missing) == n_folders:
            raise FileNotFoundError(
                f"ROCO: no split folder with captions.txt in {self.base_dir}. Looked for: {', '.join(missing)}."
            )
        if missing:
            warnings.warn(f"ROCO: these folders or files are missing, so they give no rows: {', '.join(missing)}.",
                          UserWarning, stacklevel=3)  # stacklevel 3 is the caller of harmonize()
        return pd.DataFrame(rows)
