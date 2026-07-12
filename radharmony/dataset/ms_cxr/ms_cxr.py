"""MONAI dataset for the MS-CXR Local Alignment phrase grounding benchmark.

1,047 chest X-rays from MIMIC-CXR-JPG with 1,448 (phrase, bounding box)
annotations across 8 pathology categories.

``base_image_dir`` must point at the MIMIC-CXR-JPG 2.0.0 root (the directory
containing ``files/``).  ``image_path`` in the harmonized DataFrame is
relative to this root, e.g.
``files/p10/p10233088/s54276838/675d792f-....jpg``.
"""

import os

import pandas as pd
import torch

from radharmony.harmonizer.ms_cxr import MSCXRHarmonizer
from radharmony.registry import register_dataset
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.dataset.transforms import RadiologyTransform2D


_LABEL_COLS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "lung_opacity", "pleural_effusion", "pneumonia", "pneumothorax",
]


@register_dataset("ms_cxr")
class MSCXRDataset(BaseRadiologicalDataset):
    """PyTorch/MONAI dataset for the MS-CXR phrase grounding benchmark.

    Args:
        base_image_dir: MIMIC-CXR-JPG 2.0.0 root (contains ``files/``).
            Required unless ``harmonizer_path`` or ``harmonized_df`` provided.
        csv_path: Path to ``MS_CXR_Local_Alignment_v1.1.0.csv``.
        transform: MONAI Compose transform.  Defaults to standard 2-D 224 px.
        cache_dir: MONAI PersistentDataset cache directory.
        output_cls: Yield 8 binary finding labels under ``"cls"``.
        output_bbox: Yield bounding boxes under ``"bbox"`` and ``"bbox_labels"``.
        dtype: Output tensor dtype.  Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = MSCXRHarmonizer

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
            _h = MSCXRHarmonizer.load_from_saved(harmonizer_path)
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

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_bbox:
                output_keys.add("bbox")
            transform = RadiologyTransform2D(
                img_size=224,
                output_keys=output_keys,
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
        h = MSCXRHarmonizer(
            csv_path=self._csv_path,
            mimic_base_dir=self.base_image_dir,
        )
        return h.harmonize()
