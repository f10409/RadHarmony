"""MONAI datasets for the official MIMIC-CXR (DICOM) train, validate and test splits."""

import pandas as pd

from radharmony.harmonizer import (
    MIMICCXRTestHarmonizer,
    MIMICCXRTrainHarmonizer,
    MIMICCXRValidateHarmonizer,
)
from radharmony.registry import register_dataset
from .mimic_cxr import MIMICCXRDataset


class _MIMICCXRSplitDataset(MIMICCXRDataset):
    """MIMIC-CXR (DICOM) dataset for one official split (see the subclasses).

    The arguments are the same as :class:`MIMICCXRDataset`, plus:

    Args:
        split_csv_path: Path to ``mimic-cxr-2.0.0-split.csv`` (in the
            MIMIC-CXR-JPG release). If None, the harmonizer looks next to
            ``csv_path`` and ``label_csv_path``.
    """

    def __init__(self, base_image_dir: str = None, *, split_csv_path: str = None,
                 harmonizer_path: str = None, **kwargs):
        # The parent loads a saved file with its own class, which fails for a split class.
        if harmonizer_path is not None and base_image_dir is None:
            base_image_dir = self._HARMONIZER_CLS.load_from_saved(harmonizer_path).report_base_dir
        self._split_csv_path = split_csv_path
        super().__init__(base_image_dir=base_image_dir, harmonizer_path=harmonizer_path, **kwargs)

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._HARMONIZER_CLS(
            csv_path=self._csv_path,
            label_csv_path=self._label_csv_path,
            dicom_base_dir=self._dicom_base_dir,
            report_csv_path=self._report_csv_path,
            report_base_dir=self.base_image_dir,
            split_csv_path=self._split_csv_path,
        ).harmonize(drop_uncertain=self._drop_uncertain, not_mentioned=self._not_mentioned)


@register_dataset("mimic_cxr_train")
class MIMICCXRTrainDataset(_MIMICCXRSplitDataset):
    """MIMIC-CXR (DICOM) official train split. No patient is in validate or test."""

    _HARMONIZER_CLS = MIMICCXRTrainHarmonizer


@register_dataset("mimic_cxr_validate")
class MIMICCXRValidateDataset(_MIMICCXRSplitDataset):
    """MIMIC-CXR (DICOM) official validate split. No patient is in train or test."""

    _HARMONIZER_CLS = MIMICCXRValidateHarmonizer


@register_dataset("mimic_cxr_test")
class MIMICCXRTestDataset(_MIMICCXRSplitDataset):
    """MIMIC-CXR (DICOM) official test split, with CheXpert labeler labels."""

    _HARMONIZER_CLS = MIMICCXRTestHarmonizer
