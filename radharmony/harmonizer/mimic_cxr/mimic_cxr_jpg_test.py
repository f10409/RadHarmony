"""Harmonizer for the 687-study MIMIC-CXR-JPG independently-labeled test set."""

import pandas as pd

from .mimic_cxr_jpg import MIMICCXRJPGHarmonizer


class MIMICCXRJPGTestHarmonizer(MIMICCXRJPGHarmonizer):
    """MIMIC-CXR-JPG harmonizer for the 687-study labeled test set.

    Uses the independently-labeled CSV ``mimic-cxr-2.1.0-test-set-labeled.csv``
    (PhysioNet) instead of ``mimic-cxr-2.0.0-chexpert.csv``.  The test CSV has
    only ``study_id`` (no ``subject_id``) and uses ``"Airspace Opacity"`` for
    what the standard CheXpert labels call ``"Lung Opacity"``; both quirks
    are normalised here so the harmonized DataFrame exposes the same
    LABEL_COLS as the train variant.

    Unlike the train harmonizer, this defaults to ``drop_uncertain=False`` —
    a fixed evaluation set should keep every labeled study so test metrics
    match the published numbers.  ``-1`` (uncertain) values are preserved
    as-is in the harmonized DataFrame; downstream code can apply U-Zeros /
    U-Ones / U-Ignore depending on the convention being reproduced.

    Args:
        csv_path: Path to ``mimic-cxr-2.0.0-metadata.csv``.
        label_csv_path: Path to ``mimic-cxr-2.1.0-test-set-labeled.csv``.
        base_image_dir: MIMIC-CXR-JPG ``files/`` root.
    """

    LABEL_JOIN_COLS = ["study_id"]

    def harmonize(self, *args, drop_uncertain: bool = False, **kwargs):
        return super().harmonize(*args, drop_uncertain=drop_uncertain, **kwargs)

    def _preprocess_label_df(self, label_df: pd.DataFrame) -> pd.DataFrame:
        # The test CSV uses "Airspace Opacity"; rename to the canonical
        # "Lung Opacity" used by the standard CheXpert label set.
        label_df = label_df.rename(columns={"Airspace Opacity": "Lung Opacity"})
        # Blanks (not-mentioned) → 0 (negative); preserve -1 (uncertain).
        # This differs from the train harmonizer, which converts -1 → NaN
        # so it can be dropped — for test-set evaluation we keep -1 visible.
        label_df = label_df.fillna(0.0)
        if self.drop_uncertain:
            # Opt-in: drop any row carrying at least one -1 across labels.
            label_cols = [c for c in label_df.columns if c != "study_id"]
            label_df = label_df[~(label_df[label_cols] == -1).any(axis=1)]
        label_df["study_id"] = label_df["study_id"].astype(str)
        return label_df
