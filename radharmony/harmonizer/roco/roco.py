"""Harmonizer for the ROCO (Radiology Objects in COntext) dataset.

ROCO is a large-scale multimodal captioning dataset sourced from PubMed Central.
Each entry pairs a radiology (or non-radiology) image with its figure caption,
plus UMLS keywords, CUIs, and semantic types.

**Source:** https://github.com/razorx89/roco-dataset

**Layout after running ``scripts/fetch.py``:**

    <base_dir>/
      data/
        train/
          radiology/
            captions.txt
            keywords.txt
            cuis.txt
            semtypes.txt
            images/
              ROCO_XXXXX.jpg
          non-radiology/   (same structure)
        validation/  (same)
        test/        (same)

**captions.txt format:**  ``ROCO_XXXXX\\t<caption text>``

**keywords.txt format:**  ``ROCO_XXXXX\\t\\t<kw1>\\t<kw2>\\t...``  (first field blank)

**Harmonized columns (required VQA schema):**
  patient_id, study_id, question_id, image_path, question, answer

``question`` is always the empty string (this is a captioning dataset).
``answer`` is the caption text.

**Extra columns:**
  keywords   — tab-joined keyword string
  split      — "train" / "validation" / "test"
  subset     — "radiology" / "non-radiology"
"""

import os

import pandas as pd

from ..base_vqa import BaseVQAHarmonizer


_SPLITS   = ("train", "validation", "test")
_SUBSETS  = ("radiology", "non-radiology")


def _parse_tsv(path: str, min_fields: int = 2) -> list[tuple]:
    """Read a ROCO tab-separated file, return list of (id, rest_of_fields)."""
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
        base_dir: Root of the roco-dataset clone or download tree — the
            directory that contains the ``data/`` subdirectory.
        splits: Which splits to include.  Defaults to all three:
            ``["train", "validation", "test"]``.
        radiology_only: When ``True`` (default), include only the
            ``radiology/`` subset; set to ``False`` to also include
            ``non-radiology/`` images.
        image_subdir: Name of the image subdirectory inside each split/subset
            folder (default ``"images"``).  Must match the ``--subdir``
            argument passed to ``scripts/fetch.py``.
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
        if self.base_dir is not None:
            snap["base_dir"] = self.base_dir
        if self.splits != list(_SPLITS):
            snap["splits"] = self.splits
        if not self.radiology_only:
            snap["radiology_only"] = False
        if self.image_subdir != "images":
            snap["image_subdir"] = self.image_subdir
        return snap

    def harmonize(self) -> pd.DataFrame:
        if self.base_dir is None:
            raise ValueError("base_dir is required to harmonize from scratch.")

        subsets = ("radiology",) if self.radiology_only else _SUBSETS

        rows = []
        for split in self.splits:
            for subset in subsets:
                folder = os.path.join(self.base_dir, "data", split, subset)
                if not os.path.isdir(folder):
                    continue

                cap_path = os.path.join(folder, "captions.txt")
                kw_path  = os.path.join(folder, "keywords.txt")

                # Load keywords: id → joined keyword string
                kw_map: dict[str, str] = {}
                for parts in _parse_tsv(kw_path, min_fields=1):
                    roco_id = parts[0].strip()
                    # keywords start at index 2 (index 1 is always blank)
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

        self.df = pd.DataFrame(rows).reset_index(drop=True)
        return self._select_harmonized_columns(self.df)
