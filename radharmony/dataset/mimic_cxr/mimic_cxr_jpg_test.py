"""MONAI dataset for the 687-study MIMIC-CXR-JPG independently-labeled test set."""

from radharmony.harmonizer import MIMICCXRJPGTestHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from .mimic_cxr_jpg import MIMICCXRJPGDataset


@register_dataset("mimic_cxr_jpg_test")
class MIMICCXRJPGTestDataset(MIMICCXRJPGDataset):
    """MIMIC-CXR-JPG labeled test set — strict subset of ``MIMICCXRJPGDataset``.

    Uses ``mimic-cxr-2.1.0-test-set-labeled.csv`` (687 studies) instead of
    the standard ``mimic-cxr-2.0.0-chexpert.csv``.  Same image layout as
    the parent class; only the label CSV and join keys differ.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})
    _HARMONIZER_CLS = MIMICCXRJPGTestHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_csv_path: str = None,
        drop_uncertain: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        **kwargs,
    ):
        if (
            base_image_dir
            and label_csv_path is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            label_csv_path = (
                infer_path(
                    base_image_dir,
                    "mimic-cxr-2.1.0-test-set-labeled.csv",
                    user_path="",
                )
                or None
            )

        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=csv_path,
            label_csv_path=label_csv_path,
            drop_uncertain=drop_uncertain,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
            **kwargs,
        )
        # Upgrade harmonizer to the test-specific subclass for save/load
        # round-trip identity.
        if hasattr(self, "_harmonizer") and not isinstance(
            self._harmonizer, MIMICCXRJPGTestHarmonizer
        ):
            self._harmonizer = MIMICCXRJPGTestHarmonizer(
                csv_path=self._harmonizer.csv_path,
                label_csv_path=self._harmonizer.label_csv_path,
                base_image_dir=self._harmonizer.base_image_dir,
            )
