"""MONAI dataset for the RSNA 2024 Lumbar Spine test split (images only)."""

import warnings

import torch

from radharmony.harmonizer import RSNA2024LumbarSpineTestHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from .rsna_2024_lumbar_spine import RSNA2024LumbarSpineDataset


@register_dataset("rsna_2024_lumbar_spine_test")
class RSNA2024LumbarSpineTestDataset(RSNA2024LumbarSpineDataset):
    """Test split of RSNA 2024 Lumbar Spine: images only, no labels.

    Reads ``test_series_descriptions.csv`` (``study_id``, ``series_id``,
    ``series_description``).  ``output_cls`` and ``output_bbox`` are not
    supported (no labels, no coordinate CSV).

    Args:
        base_image_dir: Root of the test DICOM tree, typically
            ``<competition_root>/test_images/``.
        series_description_csv_path: Path to ``test_series_descriptions.csv``.
            Auto-inferred from ``base_image_dir`` (and its parent) when None.
        All other args are identical to :class:`RSNA2024LumbarSpineDataset`.
    """

    _TRAIN_FILES_REQUIRED = False  # this class reads only the test files
    SUPPORTED_OUTPUTS: frozenset = frozenset()  # no labels and no coordinates (D-11)
    _HARMONIZER_CLS = RSNA2024LumbarSpineTestHarmonizer  # the saved test harmonizer loads (V-H3-21)

    def __init__(
        self,
        base_image_dir=None,
        series_description_csv_path=None,
        transform=None,
        cache_dir="./cache",
        output_cls=False,
        output_mask=False,
        output_report=False,
        output_reg=False,
        output_bbox=False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path=None,
        series_filter=None,
        dtype=torch.bfloat16,
    ):
        for flag, name in [(output_cls, "output_cls"), (output_bbox, "output_bbox")]:
            if flag:
                warnings.warn(
                    f"{name}=True is not supported by RSNA2024LumbarSpineTestDataset "
                    "(test split has no labels); ignoring.",
                    stacklevel=2,
                )

        # Pre-infer the test CSV so we can store it before super() overwrites paths.
        _test_desc_csv = series_description_csv_path
        if (
            harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
            and base_image_dir is not None
        ):
            _test_desc_csv = infer_path(
                base_image_dir, "test_series_descriptions.csv", user_path=series_description_csv_path or "",
                required=True, arg="series_description_csv_path",
            )

        # Call super. It infers train paths (irrelevant, we override
        # _get_harmonized_df below).  Pass output_cls=False / output_bbox=False.
        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=False,
            output_mask=output_mask,
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=False,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
            series_filter=series_filter,
            dtype=dtype,
        )

        self._test_series_description_csv_path = _test_desc_csv

    def _get_harmonized_df(self):
        df = self._try_resolve_preset_harmonized()
        if df is None:
            df = RSNA2024LumbarSpineTestHarmonizer(
                series_description_csv_path=self._test_series_description_csv_path,
                base_image_dir=self.base_image_dir,
            ).harmonize()
        # The filter also applies to a given table, as in the train class (V-D-32).
        if self._series_filter:
            df = df[df["view_position"] == self._series_filter].reset_index(drop=True)
        return df
