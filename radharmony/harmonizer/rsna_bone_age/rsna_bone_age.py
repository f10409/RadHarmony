"""Harmonizer for the 2017 RSNA Pediatric Bone Age Challenge dataset.

Use the primary RSNA release (Halabi et al., Radiology 2018), not Kaggle mirrors. A mirror
can differ from the release. Train and val are separate downloads, and each has its own
``base_image_dir``. Each image has one regression target, ``age_months``. There are no
class labels. ``patient_id`` and ``study_id`` are the image id.

The val CSV has other column names than the train CSV. The harmonizer renames them, so both
splits give ``age_months`` and ``sex_female``. ``sex`` is "M" or "F", made from ``male``
(D-26). The test split has its own class, :class:`RSNABoneAgeTestHarmonizer`, with no
``age_months`` (D-25).

Layout and columns: ``docs/wiki/reference/datasets/ct-mri-other/rsna_bone_age.md``.
"""

import os

import pandas as pd

from ..base import BaseHarmonizer


#: Settings for each split. The keys are the ``split`` values.
_SPLIT_CONFIGS = {
    "train": {
        "csv_filename_variants": ("train.csv",),
        # The CSV is one level above base_image_dir.
        "csv_parent_subdir": "..",
        # base_image_dir is the image folder, so image_path has no subfolder.
        "image_subdirs": (),
        # Source column names.
        "id_col": "id",
        "age_col": "boneage",
        "male_col": "male",
    },
    "val": {
        "csv_filename_variants": ("Validation Dataset.csv",),
        # The CSV is inside base_image_dir.
        "csv_parent_subdir": "",
        # The val ZIP has two image subfolders.
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

    Train and val are separate downloads. Each has its own ``base_image_dir``.

    Args:
        base_image_dir: Image folder of the split. For train, use
            ``boneage-training-dataset/`` (flat folder of ``<id>.png``). For val, use
            ``Bone Age Validation Set/`` (the CSV and the two ``boneage-validation-dataset-{1,2}/``
            subfolders).
        csv_path: Path of the CSV. If ``None``, the harmonizer looks for it next to
            ``base_image_dir`` (train) or inside it (val).
        split: ``"train"`` (default) or ``"val"``. The official release has no test labels.
    """

    LABEL_COLS: list = []
    REG_COLS = ["age_months"]
    EXTRA_OUTPUT_COLS = ["sex_female"]
    #: "M" or "F" from the ``male`` flag. The base step checks it (D-26).
    SEX_SOURCE_COL = "sex"

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
                "For the test images (no labels), use RSNABoneAgeTestHarmonizer."
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

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["id"].astype(str)

    def _build_image_path(self) -> None:
        """Set ``image_path`` relative to ``base_image_dir``.

        Train: a bare ``<id>.png``. Val: ``<subfolder>/<id>.png``, found by a scan of both
        subfolders.
        """
        cfg = _SPLIT_CONFIGS[self.split]
        subdirs = cfg["image_subdirs"]
        if not subdirs:
            self.df["image_path"] = self.df["id"].astype(str) + ".png"
            return
        if len(subdirs) == 1:
            subdir = subdirs[0]
            self.df["image_path"] = (
                subdir + "/" + self.df["id"].astype(str) + ".png"
            )
            return

        # Map each id to its path once.
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
        """Build the target ``age_months``, ``sex_female`` and ``sex`` ("M" or "F", D-26)."""
        if "boneage" in self.df.columns:
            self.df["age_months"] = self.df["boneage"].astype("int32")
        if "male" in self.df.columns:
            male = self.df["male"].astype(bool)
            self.df["sex_female"] = (~male).astype("int8")
            # The source has a True or False flag, not a sex word.
            self.df["sex"] = male.map({True: "M", False: "F"})

    def _read_source(self) -> pd.DataFrame:
        """Read the CSV of the split and rename its columns to the train names."""
        df = pd.read_csv(self.csv_path)

        cfg = _SPLIT_CONFIGS[self.split]
        # The build steps use the train column names for both splits.
        renames = {}
        if cfg["id_col"] != "id":
            renames[cfg["id_col"]] = "id"
        if cfg["age_col"] != "boneage":
            renames[cfg["age_col"]] = "boneage"
        if cfg["male_col"] != "male":
            renames[cfg["male_col"]] = "male"
        if renames:
            df = df.rename(columns=renames)
        return df
