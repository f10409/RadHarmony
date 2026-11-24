"""Harmonizer for CheXphoto (Phillips et al., ML4H 2020, arXiv:2007.06199).

Phone photographs and synthetic transforms of CheXpert x-rays. Labels come
from CheXpert unchanged (14 observations, 1/0/-1/blank). The CheXpert code
runs on this CSV without an error but gives a wrong table. Three things differ
(checked on the Redivis v1.0 download):

1. Column names use underscores (``AP_PA``, ``No_Finding``). Without
   ``COL_MAP`` the multi-word label columns do not match and are dropped.
2. ``patient_id`` is not at a fixed position. CheXpert uses
   ``Path.split("/")[2]``. CheXphoto has two more segments, so that index is
   the subset and every row gets ``patient_id="synthetic"``. The code takes
   the ``patientNNNNN`` and ``studyNNN`` segments with a regex.
3. v1.0 has no ``film`` subset, although the paper describes one.
   Train has natural/{iphone,nokia} and
   synthetic/{digital,photographic}.
   Valid has natural/oneplus and synthetic/{digital,photographic}.

``Path`` starts with a ``CheXphoto-v1.0/<split>/`` prefix that is not in the
download tree. The code removes the first two segments, as CheXpert does. All
32,521 train and 702 valid images resolve. The test split is withheld by the
authors (CodaLab only).

``harmonize(drop_uncertain=True)`` drops rows with a ``-1`` label. Train then
has 20,034 rows and 2,279 patients. valid has no ``-1`` label.
More: ``docs/wiki/reference/datasets/cxr-labels/chexphoto.md``.
"""
from __future__ import annotations

import os

from ..chexpert.chexpert import CheXpertHarmonizer


class CheXphotoHarmonizer(CheXpertHarmonizer):
    """Harmonizer for CheXphoto.

    Args:
        csv_path: ``train.csv`` or ``valid.csv``.
        base_image_dir: split root containing ``natural/`` and ``synthetic/``.
        subset: ``all`` (default), ``natural`` or ``synthetic``. It filters on
            the first path segment after the removed prefix.
    """

    # CheXphoto's CSV header, mapped onto the names CheXpertHarmonizer expects.
    COL_MAP = {
        "Frontal_Lateral": "Frontal/Lateral",
        "AP_PA": "AP/PA",
        "No_Finding": "No Finding",
        "Enlarged_Cardiomediastinum": "Enlarged Cardiomediastinum",
        "Lung_Opacity": "Lung Opacity",
        "Lung_Lesion": "Lung Lesion",
        "Pleural_Effusion": "Pleural Effusion",
        "Pleural_Other": "Pleural Other",
        "Support_Devices": "Support Devices",
    }
    EXTRA_OUTPUT_COLS = ["subset", "capture_method"]

    # The same patient appears in natural and synthetic images of one x-ray.
    # Split by patient_id, never by row.

    def __init__(self, csv_path: str, base_image_dir: str = None,
                 subset: str = "all"):
        super().__init__(csv_path=csv_path, base_image_dir=base_image_dir)
        self.subset = subset

    def _harmonizer_init_snapshot(self) -> dict:
        """Add ``subset``, so a loaded object harmonizes the same way (D-08)."""
        snap = super()._harmonizer_init_snapshot()
        snap["subset"] = self.subset
        return snap

    @classmethod
    def _split_of_saved_init(cls, init: dict):
        """Return the split that the saved ``csv_path`` names (``train.csv`` or ``valid.csv``), or None.

        A split class uses it to load a file of this class (D-09).
        """
        name = os.path.basename(str(init.get("csv_path") or ""))
        return {"train.csv": "train", "valid.csv": "valid"}.get(name)

    def _build_patient_id(self) -> None:
        """Rename the columns. This is the first hook after the base reads the CSV.

        ``self.df`` does not exist in ``__init__``, so the rename cannot go there.
        """
        self.df = self.df.rename(columns=self.COL_MAP)
        seg = self.df["Path"].str.split("/")
        self.df["subset"] = seg.str[2]              # natural | synthetic
        self.df["capture_method"] = seg.str[3]      # iphone | nokia | oneplus |
                                                    # digital | photographic
        if self.subset != "all":
            self.df = self.df[self.df["subset"] == self.subset].reset_index(drop=True)
        # CheXpert uses Path.split("/")[2]. Here that index is the subset, so
        # find the patientNNNNN segment instead.
        self.df["patient_id"] = self.df["Path"].str.extract(
            r"(patient\d+)", expand=False)

    def _build_study_id(self) -> None:
        self.df["study_id"] = (
            self.df["Path"].str.extract(r"(patient\d+)", expand=False) + "_" +
            self.df["Path"].str.extract(r"(study\d+)", expand=False))
