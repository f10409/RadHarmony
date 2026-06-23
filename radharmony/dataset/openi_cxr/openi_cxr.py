"""MONAI dataset for the OpenI Indiana University Chest X-Ray dataset (IU X-Ray).

3,955 radiology reports with 7,470 paired PNG images from the Indiana
University hospital network.

``base_image_dir`` must point at the **dataset root** — the directory
containing ``ecgen-radiology/`` (XML reports) and ``images/`` (PNG files).
``image_path`` in the harmonized DataFrame is relative to this root,
e.g. ``images/CXR1_1_IM-0001-3001.png``.
"""

import os

import pandas as pd
import torch

from radharmony.harmonizer.openi_cxr import OpenICXRHarmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


_LABEL_COLS = [
    "no_finding", "cardiomegaly", "edema", "atelectasis",
    "consolidation", "pleural_effusion", "pneumothorax", "support_devices",
]


@register_dataset("openi_cxr")
class OpenICXRDataset(BaseRadiologicalDataset):
    """PyTorch/MONAI dataset for the OpenI IU Chest X-Ray dataset.

    Args:
        base_image_dir: Dataset root directory (contains ``ecgen-radiology/``
            and ``images/``), e.g. ``.../OpenI-IU-CXR/``.  Required unless
            ``harmonizer_path`` or ``harmonized_df`` is provided.
        transform: MONAI Compose transform.  Defaults to the standard 2-D
            224 px pipeline.
        cache_dir: MONAI PersistentDataset cache directory.  ``None`` disables
            caching.
        output_cls: Yield 8 binary CheXpert-compatible labels under ``"cls"``.
        output_report: Yield full report text (findings + impression) under
            ``"report"``.
        output_mask / output_bbox: Not supported; ignored with a warning.
        dtype: Output tensor dtype.  Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "report"})
    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = OpenICXRHarmonizer

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
            _h = OpenICXRHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the path to the OpenI dataset root (the directory containing "
                "ecgen-radiology/ and images/)."
            )

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_report:
                output_keys.add("report")
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
        h = OpenICXRHarmonizer(base_dir=self.base_image_dir)
        return h.harmonize()
