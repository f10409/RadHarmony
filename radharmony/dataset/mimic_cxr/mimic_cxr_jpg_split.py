"""MONAI datasets for the official MIMIC-CXR-JPG train, validate and test splits (split file)."""

from radharmony.harmonizer import (
    MIMICCXRJPGTestCheXpertHarmonizer,
    MIMICCXRJPGTrainHarmonizer,
    MIMICCXRJPGValidateHarmonizer,
)
from radharmony.registry import register_dataset
from .mimic_cxr_jpg import MIMICCXRJPGDataset


class _MIMICCXRJPGSplitDataset(MIMICCXRJPGDataset):
    """MIMIC-CXR-JPG dataset for one official split (see the subclasses).

    The arguments are the same as :class:`MIMICCXRJPGDataset`, plus:

    Args:
        split_csv_path: Path to ``mimic-cxr-2.0.0-split.csv``. If None, the
            harmonizer looks next to ``csv_path`` and ``label_csv_path``.
    """

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_csv_path: str = None,
        *,
        split_csv_path: str = None,
        harmonizer_path: str = None,
        **kwargs,
    ):
        # The parent loads a saved file with its own class, which fails for a split class.
        if harmonizer_path is not None and base_image_dir is None:
            base_image_dir = self._HARMONIZER_CLS.load_from_saved(harmonizer_path).base_image_dir
        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=csv_path,
            label_csv_path=label_csv_path,
            harmonizer_path=harmonizer_path,
            **kwargs,
        )
        if hasattr(self, "_harmonizer"):
            h = self._harmonizer
            self._harmonizer = self._HARMONIZER_CLS(
                csv_path=h.csv_path,
                label_csv_path=h.label_csv_path,
                base_image_dir=h.base_image_dir,
                split_csv_path=split_csv_path,
            )


@register_dataset("mimic_cxr_jpg_train")
class MIMICCXRJPGTrainDataset(_MIMICCXRJPGSplitDataset):
    """MIMIC-CXR-JPG official train split, with CheXpert labeler labels.

    Use it to train. Use :class:`MIMICCXRJPGValidateDataset` to tune and
    :class:`MIMICCXRJPGTestDataset` to test. No train patient is in the
    other two.
    """

    _HARMONIZER_CLS = MIMICCXRJPGTrainHarmonizer


@register_dataset("mimic_cxr_jpg_validate")
class MIMICCXRJPGValidateDataset(_MIMICCXRJPGSplitDataset):
    """MIMIC-CXR-JPG official validate split, with CheXpert labeler labels.

    3 patients (9 studies) of this split are also in the labeled test set
    (:class:`MIMICCXRJPGTestDataset`). The class keeps them, as the split
    file says (D-34).
    """

    _HARMONIZER_CLS = MIMICCXRJPGValidateHarmonizer


@register_dataset("mimic_cxr_jpg_test_chexpert")
class MIMICCXRJPGTestCheXpertDataset(_MIMICCXRJPGSplitDataset):
    """MIMIC-CXR-JPG official test split of the split file, with CheXpert labeler labels.

    It has the same studies and labels as the DICOM key ``mimic_cxr_test``. Use
    :class:`MIMICCXRJPGTestDataset` for the radiologist labels of version 2.1.0
    (D-09).
    """

    _HARMONIZER_CLS = MIMICCXRJPGTestCheXpertHarmonizer
