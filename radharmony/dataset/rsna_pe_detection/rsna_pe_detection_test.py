"""MONAI dataset for the RSNA PE Detection test split (images only)."""

import warnings

import pandas as pd
import torch

from radharmony.harmonizer import RSNAPEDetectionTestHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from .rsna_pe_detection import RSNAPEDetectionDataset


@register_dataset("rsna_pe_detection_test")
class RSNAPEDetectionTestDataset(RSNAPEDetectionDataset):
    """Test split of RSNA PE Detection — images only, no labels.

    Reads ``test.csv`` (``StudyInstanceUID``, ``SeriesInstanceUID``,
    ``SOPInstanceUID``), deduplicates to one row per study.  No label columns.

    Args:
        base_image_dir: Root of the test DICOM tree — typically
            ``<competition_root>/test/``.
        csv_path: Path to ``test.csv``.  Auto-inferred when None.
        All other args identical to :class:`RSNAPEDetectionDataset`.
    """

    def __init__(
        self,
        base_image_dir=None,
        csv_path=None,
        transform=None,
        cache_dir="./cache",
        hu_window=(-1000, 1000),
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
                    f"{name}=True is not supported by RSNAPEDetectionTestDataset "
                    "(test split has no labels); ignoring.",
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
        # Override whatever self._csv_path the parent set via infer_path.
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = _test_csv

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = RSNAPEDetectionTestHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
