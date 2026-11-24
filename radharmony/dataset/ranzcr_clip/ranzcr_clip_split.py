"""MONAI datasets for the RANZCR CLiP labeled train rows and unlabeled test images."""

import warnings

import pandas as pd
import torch

from radharmony.harmonizer import RANZCRClipTestHarmonizer, RANZCRClipTrainHarmonizer
from radharmony.harmonizer._split import _OneSplitDatasetMixin
from radharmony.registry import register_dataset
from .ranzcr_clip import RANZCRClipDataset


@register_dataset("ranzcr_clip_train")
class RANZCRClipTrainDataset(_OneSplitDatasetMixin, RANZCRClipDataset):
    """RANZCR CLiP labeled rows of ``train.csv`` only (30,083 images, 3,255 patients).

    The arguments are the same as :class:`RANZCRClipDataset`, but there is
    no ``include_test_split``: this class never gives the test images.
    """

    _HARMONIZER_CLS = RANZCRClipTrainHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        annotations_csv_path: str = None,
        mask_output_dir: str = None,
        mask_num_cores: int = 1,
        mask_line_thickness: int = 15,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        # The parent loads a saved file with its own class, which fails for a split class.
        if harmonizer_path is not None and base_image_dir is None:
            base_image_dir = self._HARMONIZER_CLS.load_from_saved(harmonizer_path).base_image_dir
        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=csv_path,
            annotations_csv_path=annotations_csv_path,
            mask_output_dir=mask_output_dir,
            mask_num_cores=mask_num_cores,
            mask_line_thickness=mask_line_thickness,
            include_test_split=False,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
            dtype=dtype,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._HARMONIZER_CLS(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            annotations_csv_path=self._annotations_csv_path,
            mask_line_thickness=self._mask_line_thickness,
        ).harmonize(mask_output_dir=self._mask_output_dir, mask_num_cores=self._mask_num_cores)


@register_dataset("ranzcr_clip_test")
class RANZCRClipTestDataset(_OneSplitDatasetMixin, RANZCRClipDataset):
    """RANZCR CLiP Kaggle test images (3,582), with no labels and no masks (D-25).

    The class gives only ``img``. ``output_cls`` and ``output_mask`` are
    ignored with a warning.

    Args:
        base_image_dir: Root of the Kaggle download (contains ``test/``).
        The other arguments are the same as :class:`RANZCRClipDataset`.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset()
    LABEL_COLS = []
    _HARMONIZER_CLS = RANZCRClipTestHarmonizer
    _TRAIN_FILES_REQUIRED = False  # this class reads only the test images

    def __init__(
        self,
        base_image_dir: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if output_cls or output_mask:
            warnings.warn(
                "RANZCR CLiP test images have no labels and no masks. "
                "output_cls and output_mask are ignored.",
                UserWarning,
                stacklevel=2,
            )
        if harmonizer_path is not None and base_image_dir is None:
            base_image_dir = self._HARMONIZER_CLS.load_from_saved(harmonizer_path).base_image_dir
        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=False,
            output_mask=False,
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
            dtype=dtype,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._HARMONIZER_CLS(base_image_dir=self.base_image_dir).harmonize()
