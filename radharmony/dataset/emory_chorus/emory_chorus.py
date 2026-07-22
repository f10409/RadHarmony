"""MONAI dataset for the Emory CHORUS chest-radiograph subset.

Emory CHORUS is an internal Emory dataset (OMOP-CDM clinical warehouse + a
multi-modality DICOM tree).  This dataset serves the curated X-ray subset,
filtered to **chest** radiographs by default, as an image-only cohort.

Usage::

    from radharmony.dataset import EmoryCHORUSDataset

    ds = EmoryCHORUSDataset(
        base_image_dir="/mnt/NAS4/datasets/internal/Emory_CHORUS/images_dicom_xray",
        manifest_csv_path="/mnt/NAS4/datasets/internal/Emory_CHORUS/chorus_xray_manifest.csv",
    )

The first construction scans the DICOM tree and writes ``manifest_csv_path``;
later constructions read that manifest back instead of re-scanning.

No finding labels, masks, bounding boxes, or reports are available in this
release, so only the ``img`` key is produced.  ``patient_id`` (DICOM
``PatientID``) groups a patient's studies for patient-level cross-validation.
"""

import os
import warnings

import torch

from radharmony.harmonizer.emory_chorus import EmoryCHORUSHarmonizer
from radharmony.registry import register_dataset
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.dataset.transforms import RadiologyTransform2D


@register_dataset("emory_chorus")
class EmoryCHORUSDataset(BaseRadiologicalDataset):
    """Emory CHORUS X-ray subset (chest-only by default), image-only.

    Args:
        base_image_dir: Root of the DICOM tree (the ``images_dicom_xray``
            directory holding per-patient sub-folders).
        manifest_csv_path: Path to the manifest CSV used as a scan cache.  When
            it exists it is read directly; otherwise the tree is scanned and the
            manifest written here.  When ``None``, the tree is scanned in memory
            on every construction (slow — prefer passing a path).
        chest_only: Keep only chest radiographs (the subset also contains
            abdomen, spine, skull, and extremity films).  ``True`` by default.
        num_workers: Parallel worker processes for the header scan.
        transform: MONAI Compose transform.  Defaults to the standard 2-D
            pipeline (224 px, ImageNet normalisation) with DICOM VOI-LUT /
            MONOCHROME handling from the base transforms.
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls / output_mask / output_report / output_bbox: Not supported
            (no labels/masks/boxes/reports in this release); silently ignored
            with a warning.
        harmonized_df / harmonizer / harmonizer_path: Standard harmonizer preset
            hooks; see :class:`BaseRadiologicalDataset`.
        dtype: Tensor dtype for image data (default: ``torch.bfloat16``).
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset()
    LABEL_COLS: list = []
    _HARMONIZER_CLS = EmoryCHORUSHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        manifest_csv_path: str = None,
        chest_only: bool = True,
        num_workers: int = 12,
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
        if manifest_csv_path:
            manifest_csv_path = os.path.expanduser(manifest_csv_path)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the CHORUS images_dicom_xray root directory."
            )

        for flag, name in (
            (output_cls, "output_cls"),
            (output_mask, "output_mask"),
            (output_report, "output_report"),
            (output_bbox, "output_bbox"),
        ):
            if flag:
                warnings.warn(
                    f"Emory CHORUS is image-only in this release; {name}=True "
                    "ignored.",
                    UserWarning,
                    stacklevel=2,
                )

        self._manifest_csv_path = manifest_csv_path
        self._chest_only = chest_only
        self._scan_workers = num_workers

        if transform is None:
            transform = RadiologyTransform2D(
                img_size=224,
                output_keys={"img"},
                dtype=dtype,
            ).get_transform()

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=False,
            output_mask=False,
            output_report=False,
            output_bbox=False,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        h = EmoryCHORUSHarmonizer(
            base_image_dir=self.base_image_dir,
            csv_path=self._manifest_csv_path,
            chest_only=self._chest_only,
            num_workers=self._scan_workers,
        )
        return h.harmonize()
