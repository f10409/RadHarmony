"""MONAI dataset for the RSNA Pneumonia Detection Challenge (Kaggle Stage 2).

Wraps :class:`RSNAPneumoniaKaggleHarmonizer` which reads the two flat CSVs
shipped with the Kaggle distribution.  Supports classification labels
(3-class) and bounding boxes for lung opacities.

Images are flat DICOMs under ``<base_image_dir>/stage_2_train_images/``:
``<base_image_dir>/stage_2_train_images/<patientId>.dcm``
(e.g. ``~/datasets/rsna-pneumonia-detection-challenge/``).
"""

import os
import warnings

import torch

import pandas as pd

from radharmony.harmonizer import RSNAPneumoniaKaggleHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("rsna_pneumonia_kaggle")
class RSNAPneumoniaKaggleDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for RSNA Pneumonia Detection Challenge (Kaggle Stage 2).

    ~30,000 frontal chest X-rays (DICOM) with 3 classes: Normal,
    No Lung Opacity / Not Normal, and Lung Opacity.  Lung Opacity
    annotations include bounding boxes (one or more per image).

    Args:
        base_image_dir: Root directory containing ``stage_2_train_images/``
            and the two CSVs (e.g. ``~/datasets/rsna-pneumonia-detection-challenge/``).
        csv_path: Path to ``stage_2_detailed_class_info.csv``. Auto-inferred
            from ``base_image_dir`` when ``None``.
        bbox_csv_path: Path to ``stage_2_train_labels.csv``. Auto-inferred
            from ``base_image_dir`` when ``None``. Required for ``output_bbox=True``.
        transform: MONAI Compose transform. Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Not supported; silently ignored.
        output_report: Not supported; silently ignored.
        output_bbox: Yield bounding boxes under key ``bbox``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = [
        "lung_opacity",
        "no_lung_opacity_/_not_normal",
        "normal",
    ]
    _HARMONIZER_CLS = RSNAPneumoniaKaggleHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        bbox_csv_path: str = None,
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
        if bbox_csv_path:
            bbox_csv_path = os.path.expanduser(bbox_csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = RSNAPneumoniaKaggleHarmonizer.load_from_saved(harmonizer_path)
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

        if output_mask:
            warnings.warn(
                "RSNA Pneumonia does not provide segmentation masks. "
                "output_mask will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            output_mask = False
        if output_report:
            warnings.warn(
                "RSNA Pneumonia does not provide radiology reports. "
                "output_report will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            output_report = False

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = infer_path(
                base_image_dir,
                "stage_2_detailed_class_info.csv",
                user_path=csv_path or "",
            ) or csv_path
            if self._csv_path is None:
                raise ValueError(
                    "csv_path is required: stage_2_detailed_class_info.csv was not "
                    "found automatically.\n"
                    "Place it in base_image_dir or pass csv_path= explicitly."
                )
            self._bbox_csv_path = infer_path(
                base_image_dir,
                "stage_2_train_labels.csv",
                user_path=bbox_csv_path or "",
            ) or bbox_csv_path
        else:
            self._csv_path = csv_path
            self._bbox_csv_path = bbox_csv_path

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
        harmonizer = RSNAPneumoniaKaggleHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            bbox_csv_path=self._bbox_csv_path,
        )
        return harmonizer.harmonize()
