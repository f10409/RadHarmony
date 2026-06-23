"""Harmonizer for the 2017 RSNA Pediatric Bone Age Challenge dataset.

Primary source: RSNA challenge archive (referenced in Halabi S, et al.
*The Pediatric Bone Age Machine Learning Challenge.* Radiology 2018,
<https://pubs.rsna.org/doi/10.1148/radiol.2018180736>).  **Do not use
Kaggle mirrors** — they are third-party re-uploads and can diverge from
the primary release.

The official release ships **train and validation as separate downloads**
— each split has its own `base_image_dir`. They do not have to share a
parent directory.

Train layout::

    <base_image_dir>/                                    # boneage-training-dataset/
      <id>.png                                           # 12,611 PNGs
    <base_image_dir>/../train.csv                        # id, boneage, male
                                                         # (sibling — auto-discovered)

Val layout (two sub-subdirs because the val ZIP is internally split)::

    <base_image_dir>/                                    # Bone Age Validation Set/
      Validation Dataset.csv                             # Image ID, Bone Age (months), male
      boneage-validation-dataset-1/                      # 800 PNGs (ids 1386–9708)
        <id>.png
      boneage-validation-dataset-2/                      # 625 PNGs (ids 10018–15612)
        <id>.png

Regression task — each image has a single continuous target, skeletal age
in months (element ``RDE123``; official spec: 0–216, actual: 1–228 for
train, 3–228 for val).  No classification labels.  One row per image;
no distinct patient column, so ``patient_id == study_id == image id``.

The validation CSV uses different column names from the training CSV;
the harmonizer normalises both to the same schema (``age_months``,
``sex_female``) so downstream code doesn't have to case-split.

Exposed columns:

- ``patient_id``, ``study_id``, ``image_path`` — core.
- ``age_months`` — regression target (integer).  Listed in :attr:`REG_COLS`.
- ``sex_female`` — optional demographic covariate (0/1) derived from
  ``male``.  Listed in :attr:`EXTRA_OUTPUT_COLS` so it survives the
  standard column-selection pass.

The official test set (200 images) ships without labels and is not
wired in.
"""

import os

import pandas as pd

from ..base import BaseHarmonizer


#: Per-split configuration.  Keys are the public ``split`` values.
_SPLIT_CONFIGS = {
    "train": {
        "csv_filename_variants": ("train.csv",),
        # CSV is a sibling of base_image_dir (one level up).
        "csv_parent_subdir": "..",
        # base_image_dir IS the image dir; no subdir prefix in image_path.
        "image_subdirs": (),
        # Map raw CSV column names → canonical names we emit.
        "id_col": "id",
        "age_col": "boneage",
        "male_col": "male",
    },
    "val": {
        "csv_filename_variants": ("Validation Dataset.csv",),
        # CSV lives directly inside base_image_dir (Bone Age Validation Set/).
        "csv_parent_subdir": "",
        # Two sibling sub-subdirs inside base_image_dir.
        "image_subdirs": (
            "boneage-validation-dataset-1",
            "boneage-validation-dataset-2",
        ),
        "id_col": "Image ID",
        "age_col": "Bone Age (months)",
        "male_col": "male",
    },
}


class RSNABoneAgeHarmonizer(BaseHarmonizer):
    """Harmonize the 2017 RSNA Pediatric Bone Age release (train or val split).

    Each split has its own self-contained ``base_image_dir``; train and val
    are separate downloads and do not share a parent directory.

    Args:
        base_image_dir: Split-specific image directory.  For ``split="train"``
            point at ``boneage-training-dataset/`` (the flat dir of
            ``<id>.png`` files extracted from the train ZIP).  For
            ``split="val"`` point at ``Bone Age Validation Set/`` (which
            contains ``Validation Dataset.csv`` plus the two
            ``boneage-validation-dataset-{1,2}/`` subdirs).
        csv_path: Optional explicit override for the CSV path.  When
            ``None``, the harmonizer searches for it relative to
            ``base_image_dir`` (sibling for train, inside for val).
        split: ``"train"`` (default) or ``"val"``.  Test split is not
            supported — the official release does not publish labels.
    """

    LABEL_COLS: list = []
    REG_COLS = ["age_months"]
    EXTRA_OUTPUT_COLS = ["sex_female"]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    SPLITS = tuple(_SPLIT_CONFIGS.keys())

    def __init__(
        self,
        base_image_dir: str,
        csv_path: str = None,
        split: str = "train",
    ):
        if split not in _SPLIT_CONFIGS:
            raise ValueError(
                f"split={split!r} is not one of {self.SPLITS}. "
                "Test labels are not publicly released for this challenge."
            )
        self.split = split
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )
        if not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            raise ValueError(
                f"base_image_dir={self.base_image_dir!r} must be an existing directory."
            )

        resolved_csv = csv_path or self._locate_csv()
        super().__init__(csv_path=os.path.expanduser(resolved_csv) if resolved_csv else None)

    def _locate_csv(self) -> str:
        """Find the split's CSV under ``base_image_dir``."""
        cfg = _SPLIT_CONFIGS[self.split]
        parent = os.path.join(self.base_image_dir, cfg["csv_parent_subdir"])
        for fname in cfg["csv_filename_variants"]:
            candidate = os.path.join(parent, fname)
            if os.path.isfile(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find any of {cfg['csv_filename_variants']} under "
            f"{parent!r} for split={self.split!r}."
        )

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        snap["split"] = self.split
        return snap

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["id"].astype(str)

    def _build_image_path(self) -> None:
        """Resolve per-image relative paths under ``base_image_dir``.

        Train: ``base_image_dir`` is the image dir itself; ``image_path``
        is a bare ``<id>.png``.
        Val: two sibling sub-subdirs inside ``base_image_dir``; we scan
        both up-front to build a ``{id: subdir}`` lookup so each row
        picks the right one.
        """
        cfg = _SPLIT_CONFIGS[self.split]
        subdirs = cfg["image_subdirs"]
        if not subdirs:
            # base_image_dir directly contains the <id>.png files.
            self.df["image_path"] = self.df["id"].astype(str) + ".png"
            return
        if len(subdirs) == 1:
            subdir = subdirs[0]
            self.df["image_path"] = (
                subdir + "/" + self.df["id"].astype(str) + ".png"
            )
            return

        # Multi-subdir case: build a map id → relpath once.
        id_to_relpath: dict[str, str] = {}
        for subdir in subdirs:
            full = os.path.join(self.base_image_dir, subdir)
            if not os.path.isdir(full):
                continue
            for fname in os.listdir(full):
                if fname.endswith(".png"):
                    stem = fname[: -len(".png")]
                    id_to_relpath[stem] = f"{subdir}/{fname}"
        self.df["image_path"] = self.df["id"].astype(str).map(id_to_relpath)

    def _build_labels(self) -> None:
        """Produce the regression target + sex covariate.

        Train and val CSVs use different column names (``boneage`` vs
        ``Bone Age (months)``; ``id`` vs ``Image ID``) — normalise to
        ``age_months`` / ``sex_female`` regardless of split.
        """
        if "boneage" in self.df.columns:
            self.df["age_months"] = self.df["boneage"].astype("int32")
        if "male" in self.df.columns:
            self.df["sex_female"] = (~self.df["male"].astype(bool)).astype("int8")

    # ------------------------------------------------------------------
    # Custom harmonize — handles CSV column rename before the base flow
    # ------------------------------------------------------------------

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read the split's CSV, normalise column names, build standard cols."""
        self._harmonized_df_override = None
        self.df = pd.read_csv(self.csv_path)

        cfg = _SPLIT_CONFIGS[self.split]
        # Rename to the train schema so downstream hooks can be split-agnostic.
        renames = {}
        if cfg["id_col"] != "id":
            renames[cfg["id_col"]] = "id"
        if cfg["age_col"] != "boneage":
            renames[cfg["age_col"]] = "boneage"
        if cfg["male_col"] != "male":
            renames[cfg["male_col"]] = "male"
        if renames:
            self.df = self.df.rename(columns=renames)

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self._build_labels()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.LABEL_COLS = sorted(
            self.LABEL_COLS, key=lambda c: c.lower().replace(" ", "_")
        )
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
