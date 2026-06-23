"""MONAI dataset for the RSNA 2022 Cervical Spine test split (images only)."""

import warnings

import pandas as pd
import torch

from radharmony.harmonizer import RSNA2022CervicalSpineTestHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from .rsna_2022_cervical_spine import RSNA2022CervicalSpineDataset


@register_dataset("rsna_2022_cervical_spine_test")
class RSNA2022CervicalSpineTestDataset(RSNA2022CervicalSpineDataset):
    """Test split of RSNA 2022 Cervical Spine — images only, no labels.

    Reads ``test.csv`` (``row_id``, ``StudyInstanceUID``, ``prediction_type``),
    deduplicates to one row per study.  No label columns; no segmentation masks.

    Args:
        base_image_dir: Root of the test DICOM tree — typically
            ``<competition_root>/test_images/``.
        csv_path: Path to ``test.csv``.  Auto-inferred when None.
        All other args identical to :class:`RSNA2022CervicalSpineDataset`.
    """

    def __init__(
        self,
        base_image_dir=None,
        csv_path=None,
        segmentation_dir=None,
        transform=None,
        cache_dir="./cache",
        hu_window=(-200, 1800),
        output_cls=False,
        output_mask=False,
        output_report=False,
        output_bbox=False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path=None,
        dtype=torch.bfloat16,
    ):
        for flag, name in [
            (output_cls, "output_cls"), (output_mask, "output_mask"),
            (output_report, "output_report"), (output_bbox, "output_bbox"),
        ]:
            if flag:
                warnings.warn(
                    f"{name}=True is not supported by RSNA2022CervicalSpineTestDataset "
                    "(test split has no labels or segmentation masks); ignoring.",
                    stacklevel=2,
                )

        # Pre-infer test.csv before super() can override with train.csv.
        _test_csv = csv_path
        if (
            harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
            and base_image_dir is not None
            and _test_csv is None
        ):
            _test_csv = infer_path(base_image_dir, "test.csv", user_path="") or None

        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=_test_csv,
            segmentation_dir=None,
            transform=transform,
            cache_dir=cache_dir,
            hu_window=hu_window,
            output_cls=False,
            output_mask=False,
            output_report=False,
            output_bbox=False,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
            dtype=dtype,
        )
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = _test_csv

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = RSNA2022CervicalSpineTestHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
