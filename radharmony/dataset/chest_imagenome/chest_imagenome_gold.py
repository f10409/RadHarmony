"""MONAI dataset for the Chest ImaGenome gold set (annotation layer over MIMIC-CXR)."""

import os

import pandas as pd
import torch

from radharmony.harmonizer import ChestImaGenomeGoldHarmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("chest_imagenome_gold")
class ChestImaGenomeGoldDataset(BaseRadiologicalDataset):
    """2-D MONAI dataset for Chest ImaGenome gold (1,000 verified images).

    Chest ImaGenome ships annotations only; the pixels come from the MIMIC-CXR
    **DICOM** tree.  Each sample carries up to ~26 anatomical-region bounding
    boxes.  Per-box **anatomy** is in ``bbox_labels``; per-box **findings** are
    in the harmonized ``bbox_findings`` column (see
    :class:`~radharmony.harmonizer.chest_imagenome.ChestImaGenomeGoldHarmonizer`).

    Args:
        base_image_dir: Root of the MIMIC-CXR DICOM file tree (children are
            ``p10/``, ``p11/`` ...), e.g. ``.../MIMIC-CXR-V2-AWS/files/``.
        annotation_dir: Chest ImaGenome release root (contains ``gold_dataset/``,
            ``utils/``), e.g. ``/data/CHEST-IMAGENOME/``.
        transform: MONAI Compose transform. Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_bbox: Yield boxes under ``bbox`` and anatomy under ``bbox_labels``
            (default ``True`` - boxes are this dataset's sole annotation).
        dtype: Output tensor dtype. Default ``torch.bfloat16``.

    Note:
        This is MIMIC-derived, PhysioNet-credentialed data - never commit the
        harmonized DataFrame, saved CSVs, or images.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"bbox"})
    SUPPORTS_BBOX_FINDINGS: bool = True
    LABEL_COLS = []
    _HARMONIZER_CLS = ChestImaGenomeGoldHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        annotation_dir: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = True,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if annotation_dir:
            annotation_dir = os.path.expanduser(annotation_dir)

        _preset = harmonizer_path is not None or harmonized_df is not None or harmonizer is not None

        if harmonizer_path is not None and base_image_dir is None:
            _h = ChestImaGenomeGoldHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_image_dir", None)

        if base_image_dir is None and not _preset:
            raise ValueError(
                "base_image_dir (MIMIC-CXR DICOM files/ root) is required when "
                "harmonizer_path is not provided."
            )
        if annotation_dir is None and not _preset:
            raise ValueError(
                "annotation_dir (Chest ImaGenome release root) is required when "
                "harmonizer_path is not provided."
            )

        self._annotation_dir = annotation_dir

        if transform is None:
            output_keys = {"img"}
            if output_bbox:
                output_keys.add("bbox")
                output_keys.add("bbox_labels")
                output_keys.add("bbox_findings")
            transform = RadiologyTransform2D(
                img_size=224, output_keys=output_keys, dtype=dtype,
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
        return ChestImaGenomeGoldHarmonizer(
            annotation_dir=self._annotation_dir,
            base_image_dir=self.base_image_dir,
        ).harmonize()
