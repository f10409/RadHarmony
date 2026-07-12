"""MONAI dataset for the EmoryCXR v2 de-identified chest radiography dataset.

EmoryCXR v2 is an internal Emory dataset (~2.4M images) with de-identified
PNGs, study-level CheXpert-aligned finding labels (14 classes, binary
0/1/NaN — no uncertain ``-1``), and de-identified free-text reports.

Usage::

    from radharmony.dataset import EmoryCXRDataset

    ds = EmoryCXRDataset(
        base_image_dir="/path/to/EmoryCXRv2/DEID_PNG",
        csv_path="/path/to/EmoryCXRv2/TABLES/metadata.csv",
        label_csv_path="/path/to/EmoryCXRv2/TABLES/finding_labels.csv",
        report_csv_path="/path/to/EmoryCXRv2/TABLES/reports.csv",
        output_cls=True,
        output_report=True,
    )

Masks and bounding boxes are not available in this dataset.
"""

import os
import warnings

import pandas as pd
import torch

from radharmony.harmonizer.emory_cxr import EmoryCXRHarmonizer
from radharmony.registry import register_dataset
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.dataset.transforms import RadiologyTransform2D


_LABEL_COLS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged_cardiomediastinum",
    "fracture",
    "lung_lesion",
    "lung_opacity",
    "no_finding",
    "pleural_effusion",
    "pleural_other",
    "pneumonia",
    "pneumothorax",
    "support_devices",
]


@register_dataset("emory_cxr")
class EmoryCXRDataset(BaseRadiologicalDataset):
    """EmoryCXR v2 PNG dataset.

    Args:
        base_image_dir: Root of the PNG image tree (the directory containing
            per-patient sub-folders, e.g.
            ``/path/to/EmoryCXRv2/DEID_PNG``).
        csv_path: Path to the image-level metadata CSV (e.g.
            ``metadata.csv``).  Auto-discovered near ``base_image_dir`` when ``None``.
        label_csv_path: Path to the study-level finding-label CSV (e.g.
            ``finding_labels.csv``).  When ``None``, no ``cls`` output is
            possible and ``output_cls`` is silently forced to ``False``.
        report_csv_path: Path to the de-identified report CSV (e.g.
            ``reports.csv``).  When ``None``, ``output_report``
            is silently forced to ``False``.
        transform: MONAI Compose transform.  Defaults to the standard 2-D
            pipeline (224 px, ImageNet normalisation).
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls: Yield 14-class binary label vector under key ``cls``.
            Requires ``label_csv_path``.
        output_report: Yield de-identified report text under key ``report``.
            Requires ``report_csv_path``.
        output_mask / output_bbox: Not supported; silently ignored with a warning.
        harmonized_df / harmonizer / harmonizer_path: Standard harmonizer
            preset hooks; see :class:`BaseRadiologicalDataset`.
        dtype: Tensor dtype for image data (default: ``torch.bfloat16``).
    """

    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = EmoryCXRHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_csv_path: str = None,
        report_csv_path: str = None,
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
        if label_csv_path:
            label_csv_path = os.path.expanduser(label_csv_path)
        if report_csv_path:
            report_csv_path = os.path.expanduser(report_csv_path)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the EmoryCXRv2 DEID_PNG root directory."
            )

        if output_cls and label_csv_path is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            warnings.warn(
                "output_cls=True but no label_csv_path provided; cls output will be all-NaN. "
                "Pass label_csv_path= to include finding labels.",
                UserWarning, stacklevel=2,
            )

        if output_report and report_csv_path is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            warnings.warn(
                "output_report=True but no report_csv_path provided; report column will be absent. "
                "Pass report_csv_path= to include de-identified reports.",
                UserWarning, stacklevel=2,
            )
            output_report = False

        if output_mask:
            warnings.warn(
                "EmoryCXR v2 does not provide segmentation masks; output_mask=True ignored.",
                UserWarning, stacklevel=2,
            )
            output_mask = False

        if output_bbox:
            warnings.warn(
                "EmoryCXR v2 does not provide bounding boxes; output_bbox=True ignored.",
                UserWarning, stacklevel=2,
            )
            output_bbox = False

        self._label_csv_path = label_csv_path
        self._report_csv_path = report_csv_path
        self._csv_path = csv_path

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
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
        h = EmoryCXRHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            label_csv_path=self._label_csv_path,
            report_csv_path=self._report_csv_path,
        )
        return h.harmonize()
