"""Harmonizer for the MIMIC-CXR-JPG independently-labeled test set (685 studies)."""

import pandas as pd

from .._labels import chexpert_codes
from .mimic_cxr_jpg import MIMICCXRJPGHarmonizer


class MIMICCXRJPGTestHarmonizer(MIMICCXRJPGHarmonizer):
    """MIMIC-CXR-JPG harmonizer for the labeled test set.

    The label file has 687 studies. 2 of them (54107786, 58592606) have no
    JPG in the metadata file, so the inner merge gives 685 studies.

    It uses the independently labeled CSV ``mimic-cxr-2.1.0-test-set-labeled.csv``
    (PhysioNet) instead of ``mimic-cxr-2.0.0-chexpert.csv``. The test CSV has
    only ``study_id`` (no ``subject_id``). It calls ``"Lung Opacity"`` by the
    name ``"Airspace Opacity"``. The code fixes both, so the table has the
    same LABEL_COLS as the train variant.

    The default is ``drop_uncertain=False``, unlike the train harmonizer. A
    fixed test set keeps every labeled study, so the metrics match the
    published numbers. ``-1`` (uncertain) values stay as they are. Later code
    can apply U-Zeros, U-Ones or U-Ignore.

    Args:
        csv_path: Path to ``mimic-cxr-2.0.0-metadata.csv``.
        label_csv_path: Path to ``mimic-cxr-2.1.0-test-set-labeled.csv``.
        base_image_dir: MIMIC-CXR-JPG ``files/`` root.
    """

    LABEL_JOIN_COLS = ["study_id"]

    def harmonize(self, *args, drop_uncertain: bool = False, **kwargs):
        return super().harmonize(*args, drop_uncertain=drop_uncertain, **kwargs)

    def _preprocess_label_df(self, label_df: pd.DataFrame) -> pd.DataFrame:
        # The test CSV says "Airspace Opacity". CheXpert says "Lung Opacity".
        label_df = label_df.rename(columns={"Airspace Opacity": "Lung Opacity"})
        label_df = chexpert_codes(label_df, self.LABEL_COLS, self.not_mentioned, self.drop_uncertain)
        label_df["study_id"] = label_df["study_id"].astype(str)
        return label_df
