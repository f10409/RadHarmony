"""MONAI dataset for the VinDr-PCXR Pediatric Chest X-ray dataset.

9,125 pediatric CXR DICOMs (7,728 train + 1,397 test) from VinMec
International Hospital, annotated by experienced radiologists with
15 image-level binary labels and bounding boxes for 37 thoracic conditions.

``base_image_dir`` must point at the **dataset root** — the directory
containing ``train/``, ``test/``, and the four CSV files.
``image_path`` in the harmonized DataFrame is relative to this root,
e.g. ``train/6cb53aff85c71b98ad13d67a131708c6.dicom``.
"""

import os

import pandas as pd
import torch

from radharmony.harmonizer.vindr_pcxr import VinDrPCXRHarmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


_LABEL_COLS = [
    "no_finding", "bronchitis", "brocho_pneumonia", "other_disease",
    "bronchiolitis", "situs_inversus", "pneumonia", "pleuro_pneumonia",
    "diagphramatic_hernia", "tuberculosis", "congenital_emphysema", "cpam",
    "hyaline_membrane_disease", "mediastinal_tumor", "lung_tumor",
]


@register_dataset("vindr_pcxr")
class VinDrPCXRDataset(BaseRadiologicalDataset):
    """PyTorch/MONAI dataset for the VinDr-PCXR Pediatric Chest X-ray dataset.

    Args:
        base_image_dir: Dataset root directory (contains ``train/``, ``test/``,
            and CSV files), e.g. ``.../VINDR-PCXR/``.  Required unless
            ``harmonizer_path`` or ``harmonized_df`` is provided.
        transform: MONAI Compose transform.  Defaults to the standard 2-D
            224 px pipeline.
        cache_dir: MONAI PersistentDataset cache directory.  ``None`` disables
            caching.
        output_cls: Yield 15 binary condition labels under ``"cls"``.
        output_bbox: Yield bounding boxes under ``"bbox"`` and class names
            under ``"bbox_labels"``.
        output_mask: Not supported; ignored with a warning.
        output_report: Not supported; ignored with a warning.
        dtype: Output tensor dtype.  Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = VinDrPCXRHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
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

        if harmonizer_path is not None and base_image_dir is None:
            _h = VinDrPCXRHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the path to the VinDr-PCXR dataset root (the directory "
                "containing train/ and test/)."
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
        h = VinDrPCXRHarmonizer(base_dir=self.base_image_dir)
        return h.harmonize()
