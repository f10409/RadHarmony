"""MONAI dataset for the RSNA 2023 Abdominal Trauma test split (images only)."""

import warnings

import pandas as pd
import torch

from radharmony.harmonizer import RSNAAbdominalTrauma2023TestHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from .rsna_abdominal_trauma_2023 import RSNAAbdominalTrauma2023Dataset


@register_dataset("rsna_abdominal_trauma_2023_test")
class RSNAAbdominalTrauma2023TestDataset(RSNAAbdominalTrauma2023Dataset):
    """Test split of RSNA 2023 Abdominal Trauma — images only, no labels.

    Reads ``test_series_meta.csv`` (``patient_id``, ``series_id``,
    ``aortic_hu``).  No patient-level label CSV, no join.  ``aortic_hu``
    is preserved as an extra output column.

    Args:
        base_image_dir: Root of the test DICOM tree — typically
            ``<competition_root>/test_images/``.
        series_meta_csv_path: Path to ``test_series_meta.csv``.
            Auto-inferred when None.
        All other args identical to :class:`RSNAAbdominalTrauma2023Dataset`.
    """

    def __init__(
        self,
        base_image_dir=None,
        csv_path=None,
        series_meta_csv_path=None,
        transform=None,
        cache_dir="./cache",
        hu_window=(-150, 250),
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
                    f"{name}=True is not supported by RSNAAbdominalTrauma2023TestDataset "
                    "(test split has no labels); ignoring.",
                    stacklevel=2,
                )

        # Pre-infer test_series_meta.csv.
        _test_meta_csv = series_meta_csv_path
        if (
            harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
            and base_image_dir is not None
            and _test_meta_csv is None
        ):
            _test_meta_csv = (
                infer_path(base_image_dir, "test_series_meta.csv", user_path="") or None
            )

        # Pass both csv_path and series_meta_csv_path pointing to the test meta CSV
        # so infer_path (user_path fast-path) sets self._series_meta_csv_path correctly.
        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=_test_meta_csv,
            series_meta_csv_path=_test_meta_csv,
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
        # Ensure the test meta CSV is stored regardless of infer_path outcome.
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._series_meta_csv_path = _test_meta_csv

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = RSNAAbdominalTrauma2023TestHarmonizer(
            series_meta_csv_path=self._series_meta_csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
