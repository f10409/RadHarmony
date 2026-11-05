"""Harmonizers for the official MIMIC-CXR-JPG train, validate and test splits (split file)."""

from .mimic_cxr_jpg import MIMICCXRJPGHarmonizer
from .mimic_split import _MIMICSplitMixin


class _MIMICCXRJPGSplitHarmonizer(_MIMICSplitMixin, MIMICCXRJPGHarmonizer):
    """MIMIC-CXR-JPG harmonizer for one official split (see the subclasses).

    Args:
        csv_path: Path to ``mimic-cxr-2.0.0-metadata.csv``.
        label_csv_path: Path to ``mimic-cxr-2.0.0-chexpert.csv``.
        base_image_dir: MIMIC-CXR-JPG ``files/`` root.
        split_csv_path: Path to ``mimic-cxr-2.0.0-split.csv``. If None, the
            class looks next to ``csv_path`` and ``label_csv_path``.
    """

    def __init__(
        self,
        csv_path: str,
        label_csv_path: str = None,
        base_image_dir: str = None,
        split_csv_path: str = None,
    ):
        super().__init__(csv_path=csv_path, label_csv_path=label_csv_path, base_image_dir=base_image_dir)
        self.split_csv_path = split_csv_path


class MIMICCXRJPGTrainHarmonizer(_MIMICCXRJPGSplitHarmonizer):
    """MIMIC-CXR-JPG official train split (``split == "train"`` in the split file).

    Labels come from ``mimic-cxr-2.0.0-chexpert.csv``. No train patient is in
    the labeled test set (:class:`MIMICCXRJPGTestHarmonizer`).
    """

    SPLIT = "train"


class MIMICCXRJPGValidateHarmonizer(_MIMICCXRJPGSplitHarmonizer):
    """MIMIC-CXR-JPG official validate split (``split == "validate"`` in the split file).

    Labels come from ``mimic-cxr-2.0.0-chexpert.csv``. 3 patients (9 studies)
    of this split are also in the labeled test set
    (:class:`MIMICCXRJPGTestHarmonizer`). The class keeps them, as the split
    file says (D-34).
    """

    SPLIT = "validate"


class MIMICCXRJPGTestCheXpertHarmonizer(_MIMICCXRJPGSplitHarmonizer):
    """MIMIC-CXR-JPG official test split (``split == "test"`` in the split file).

    Labels come from ``mimic-cxr-2.0.0-chexpert.csv`` (the CheXpert labeler),
    as for the DICOM class :class:`MIMICCXRTestHarmonizer`. The radiologist
    labels of version 2.1.0 are in :class:`MIMICCXRJPGTestHarmonizer` (D-09).
    """

    SPLIT = "test"
