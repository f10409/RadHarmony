"""MONAI dataset for the RSNA Pneumonia Detection Challenge.

Wraps the :class:`RSNAPneumoniaHarmonizer` which parses the adjudicated
JSON annotation export.  Supports classification labels (3-class) and
bounding boxes for lung opacities.

Images are DICOM files in a nested directory structure:
``<StudyUID>/<SeriesUID>/<SOPUID>.dcm`` under the base directory
(e.g. ``~/Downloads/rsna/``).
"""

import os

import pandas as pd
import torch

from radharmony.harmonizer import RSNAPneumoniaHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("rsna_pneumonia")
class RSNAPneumoniaDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for RSNA Pneumonia Detection Challenge.

    ~29,684 frontal chest X-rays (DICOM) with 3 classes:
    Normal, No Lung Opacity / Not Normal, and Lung Opacity.
    Lung Opacity annotations include bounding boxes.

    Args:
        base_image_dir: Root of the image export containing
            ``<StudyUID>/<SeriesUID>/<SOPUID>.dcm`` (e.g. ``~/Downloads/rsna/``).
        csv_path: Path to the adjudicated JSON annotation file.  Auto-inferred
            when ``None``.
        transform: MONAI Compose transform. Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Not supported (no segmentation masks); silently ignored.
        output_report: Not supported (no radiology reports); silently ignored.
        output_bbox: Yield bounding boxes under key ``bbox``.
        label_group: Label group name in the JSON.  Default ``"Calculated"``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = [
        "lung_opacity",
        "no_lung_opacity_/_not_normal",
        "normal",
    ]
    _HARMONIZER_CLS = RSNAPneumoniaHarmonizer

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
        label_group: str = "Calculated",
        dtype=torch.bfloat16,
    ):
        # Expand ~ so paths typed in Gradio or shell work correctly.
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if csv_path:
            csv_path = os.path.expanduser(csv_path)

        # Infer base_image_dir from saved harmonizer if needed.
        if harmonizer_path is not None and base_image_dir is None:
            _h = RSNAPneumoniaHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_image_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        # Auto-discover JSON annotation file near base_image_dir.
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = infer_path(
                base_image_dir,
                "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json",
                user_path=csv_path or "",
            ) or csv_path
        else:
            self._csv_path = csv_path

        self._label_group = label_group

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_bbox:
                output_keys.add("bbox")
            transform = RadiologyTransform2D(
                img_size=224, output_keys=output_keys,
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
        harmonizer = RSNAPneumoniaHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            label_group=self._label_group,
        )
        return harmonizer.harmonize()
