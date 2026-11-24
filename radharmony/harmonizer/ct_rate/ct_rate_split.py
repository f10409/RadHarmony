"""Harmonizers for the official CT-RATE train and valid splits."""

from .ct_rate import CTRATEHarmonizer


class _CTRATESplitHarmonizer(CTRATEHarmonizer):
    """CT-RATE harmonizer for one split. It keeps the volumes whose name starts with ``<SPLIT>_``.

    A label CSV of the other split has no such volume, so the class raises
    ``ValueError``. The filter is in the patient ID step.
    """

    SPLIT: str = None

    def _build_patient_id(self) -> None:
        keep = self.df["VolumeName"].astype(str).str.startswith(f"{self.SPLIT}_")
        if not keep.any():
            raise ValueError(f"{self.csv_path} has no {self.SPLIT}_ volume. Pass the {self.SPLIT} label CSV.")
        self.df = self.df[keep].reset_index(drop=True)
        super()._build_patient_id()


class CTRATETrainHarmonizer(_CTRATESplitHarmonizer):
    """CT-RATE official train split (``train_predicted_labels.csv``)."""

    SPLIT = "train"


class CTRATEValidHarmonizer(_CTRATESplitHarmonizer):
    """CT-RATE official valid split (``valid_predicted_labels.csv``)."""

    SPLIT = "valid"
