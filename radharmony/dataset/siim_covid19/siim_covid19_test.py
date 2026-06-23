"""MONAI dataset for the SIIM COVID-19 Detection Challenge — test split."""

import warnings

import torch

from radharmony.harmonizer import SIIMCOVID19TestHarmonizer
from radharmony.registry import register_dataset
from .siim_covid19 import SIIMCOVID19Dataset


@register_dataset("siim_covid19_test")
class SIIMCOVID19TestDataset(SIIMCOVID19Dataset):
    """Test split of the SIIM COVID-19 Detection Challenge — images only, no labels.

    Walks the ``test/`` DICOM tree (``study/series/sop.dcm``).
    ``output_cls`` and ``output_bbox`` are silently suppressed.

    Args:
        base_image_dir: Root of the test DICOM tree — typically
            ``<competition_root>/test/``.
        All other args identical to :class:`SIIMCOVID19Dataset`.
    """

    def __init__(
        self,
        base_image_dir=None,
        transform=None,
        cache_dir="./cache",
        output_cls=False,
        output_mask=False,
        output_report=False,
        output_bbox=False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path=None,
        dtype=torch.bfloat16,
    ):
        for flag, name in [(output_cls, "output_cls"), (output_bbox, "output_bbox")]:
            if flag:
                warnings.warn(
                    f"{name}=True is not supported by SIIMCOVID19TestDataset "
                    "(test split has no labels); ignoring.",
                    stacklevel=2,
                )

        super().__init__(
            base_image_dir=base_image_dir,
            csv_path=None,
            image_csv_path=None,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=False,
            output_mask=False,
            output_report=False,
            output_bbox=False,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
            dtype=dtype,
        )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return SIIMCOVID19TestHarmonizer(base_image_dir=self.base_image_dir).harmonize()
