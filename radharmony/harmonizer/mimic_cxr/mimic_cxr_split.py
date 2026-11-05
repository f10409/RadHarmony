"""Harmonizers for the official MIMIC-CXR (DICOM) train, validate and test splits."""

from .mimic_cxr import MIMICCXRHarmonizer
from .mimic_split import _MIMICSplitMixin


class _MIMICCXRSplitHarmonizer(_MIMICSplitMixin, MIMICCXRHarmonizer):
    """MIMIC-CXR (DICOM) harmonizer for one official split (see the subclasses).

    Labels come from ``mimic-cxr-2.0.0-chexpert.csv``. The classes keep the
    split file as it is: the 3 validate patients of the MIMIC-CXR-JPG
    labeled test set stay in validate.

    Args:
        csv_path: Path to ``cxr-record-list.csv.gz``.
        label_csv_path: Path to ``mimic-cxr-2.0.0-chexpert.csv`` (optional).
        dicom_base_dir: Root of the DICOM file tree, to read ``ViewPosition``.
        report_csv_path: Path to ``cxr-study-list.csv.gz`` (optional).
        report_base_dir: Root directory prepended to relative report paths.
        split_csv_path: Path to ``mimic-cxr-2.0.0-split.csv``. If None, the
            class looks next to ``csv_path`` and ``label_csv_path``.
    """

    def __init__(
        self,
        csv_path: str,
        label_csv_path: str = None,
        dicom_base_dir: str = None,
        report_csv_path: str = None,
        report_base_dir: str = None,
        split_csv_path: str = None,
    ):
        super().__init__(
            csv_path=csv_path,
            label_csv_path=label_csv_path,
            dicom_base_dir=dicom_base_dir,
            report_csv_path=report_csv_path,
            report_base_dir=report_base_dir,
        )
        self.split_csv_path = split_csv_path


class MIMICCXRTrainHarmonizer(_MIMICCXRSplitHarmonizer):
    """MIMIC-CXR (DICOM) official train split (``split == "train"``)."""

    SPLIT = "train"


class MIMICCXRValidateHarmonizer(_MIMICCXRSplitHarmonizer):
    """MIMIC-CXR (DICOM) official validate split (``split == "validate"``)."""

    SPLIT = "validate"


class MIMICCXRTestHarmonizer(_MIMICCXRSplitHarmonizer):
    """MIMIC-CXR (DICOM) official test split (``split == "test"``), with CheXpert labeler labels."""

    SPLIT = "test"
