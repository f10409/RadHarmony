"""Harmonizer for the RSNA Bone Age test images (200 images, no labels, D-25)."""

import os

import pandas as pd

from .rsna_bone_age import RSNABoneAgeHarmonizer


class RSNABoneAgeTestHarmonizer(RSNABoneAgeHarmonizer):
    """RSNA Bone Age test split: the 200 challenge test images, with no bone age.

    Lists the ``<id>.png`` files in ``base_image_dir`` (for example the
    Kaggle folder ``boneage-test-dataset/``). The table has no ``age_months``
    (D-25).

    Args:
        base_image_dir: The folder that holds the test PNGs.
    """

    REG_COLS = []
    LABEL_COLS = []

    def __init__(self, base_image_dir: str):
        # The parent looks for a label CSV of the split. The test split has none.
        self.split = "test"
        self.base_image_dir = os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        super(RSNABoneAgeHarmonizer, self).__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        return {"base_image_dir": self.base_image_dir}

    def _read_source(self) -> pd.DataFrame:
        """List the test PNGs, one row per image (no labels)."""
        if not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            raise ValueError(f"base_image_dir={self.base_image_dir!r} must be the folder with the test PNGs.")
        ids = sorted(f[: -len(".png")] for f in os.listdir(self.base_image_dir) if f.endswith(".png"))
        if not ids:
            raise FileNotFoundError(f"No PNG in {self.base_image_dir!r}.")
        return pd.DataFrame({"id": ids})

    def _build_image_path(self) -> None:
        # _SPLIT_CONFIGS has no "test" entry. The test folder is flat: <id>.png.
        self.df["image_path"] = self.df["id"].astype(str) + ".png"
