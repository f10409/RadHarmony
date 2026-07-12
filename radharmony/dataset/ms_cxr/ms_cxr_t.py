"""MONAI dataset for the MS-CXR-T temporal image pairs benchmark.

1,045 longitudinal chest X-ray pairs from MIMIC-CXR-JPG with progression
labels (stable / improving / worsening) for up to 5 pathology findings.

``base_image_dir`` must point at the MIMIC-CXR-JPG 2.0.0 root (contains
``files/``).  Each sample contains both ``"img"`` (current visit) and
``"previous_img"`` (prior visit) tensors when ``output_previous=True``.
"""

import os

import pandas as pd
import torch

from radharmony.harmonizer.ms_cxr_t import MSCXRTHarmonizer
from radharmony.registry import register_dataset
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.dataset.transforms import RadiologyTransform2D


@register_dataset("ms_cxr_t")
class MSCXRTDataset(BaseRadiologicalDataset):
    """PyTorch/MONAI dataset for the MS-CXR-T temporal progression benchmark.

    Args:
        base_image_dir: MIMIC-CXR-JPG 2.0.0 root (contains ``files/``).
        csv_path: Path to ``MS_CXR_T_temporal_image_classification_v1.0.0.csv``.
        transform: MONAI Compose transform.  Defaults to standard 2-D 224 px.
        cache_dir: MONAI PersistentDataset cache directory.
        output_previous: Also load the previous image into ``"previous_img"``.
        dtype: Output tensor dtype.  Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"previous_img"})
    LABEL_COLS = []
    _HARMONIZER_CLS = MSCXRTHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        output_previous: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if csv_path:
            csv_path = os.path.expanduser(csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = MSCXRTHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "mimic_base_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir (MIMIC-CXR-JPG 2.0.0 root) is required when "
                "harmonizer_path is not provided."
            )

        self._output_previous = output_previous

        if transform is None:
            transform = RadiologyTransform2D(
                img_size=224,
                output_keys={"img"},
                dtype=dtype,
            ).get_transform()

        self._csv_path = csv_path
        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        h = MSCXRTHarmonizer(
            csv_path=self._csv_path,
            mimic_base_dir=self.base_image_dir,
        )
        return h.harmonize()
