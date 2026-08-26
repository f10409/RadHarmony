"""MONAI dataset for the Chest ImaGenome silver set (full scene-graph release)."""

import os

import pandas as pd
import torch

from radharmony.harmonizer import ChestImaGenomeSilverHarmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("chest_imagenome_silver")
class ChestImaGenomeSilverDataset(BaseRadiologicalDataset):
    """2-D MONAI dataset for Chest ImaGenome silver (auto-generated scene graphs).

    The full ~240k-image auto-labeled release with official train/valid/test
    splits.  Pixels come from the MIMIC-CXR **DICOM** tree; each sample carries
    up to ~36 anatomical-region bounding boxes with per-box anatomy
    (``bbox_labels``) and per-box findings (harmonized ``bbox_findings``
    column).  See
    :class:`~radharmony.harmonizer.chest_imagenome.ChestImaGenomeSilverHarmonizer`.

    Args:
        base_image_dir: Root of the MIMIC-CXR DICOM file tree (children are
            ``p10/``, ``p11/`` ...), e.g. ``.../MIMIC-CXR-V2-AWS/files/``.
        annotation_dir: Chest ImaGenome release root (contains
            ``silver_dataset/``), e.g. ``/data/CHEST-IMAGENOME/``.
        split: One of ``"all"`` (default), ``"train"``, ``"valid"``, ``"test"``.
        transform: MONAI Compose transform. Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_bbox: Yield boxes under ``bbox`` and anatomy under ``bbox_labels``
            (default ``True`` - boxes are this dataset's sole annotation).
        dtype: Output tensor dtype. Default ``torch.bfloat16``.

    Note:
        Instantiating a full split parses the scene-graph archive and reads one
        DICOM header per image for bbox normalization - a one-time cost cached by
        the harmonizer's ``save()``.  This is MIMIC-derived, PhysioNet data -
        never commit the harmonized DataFrame, saved CSVs, or images.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"bbox"})
    SUPPORTS_BBOX_FINDINGS: bool = True
    LABEL_COLS = []
    _HARMONIZER_CLS = ChestImaGenomeSilverHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        annotation_dir: str = None,
        split: str = "all",
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
            _h = ChestImaGenomeSilverHarmonizer.load_from_saved(harmonizer_path)
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
        self._split = split

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
        return ChestImaGenomeSilverHarmonizer(
            annotation_dir=self._annotation_dir,
            base_image_dir=self.base_image_dir,
            split=self._split,
        ).harmonize()
