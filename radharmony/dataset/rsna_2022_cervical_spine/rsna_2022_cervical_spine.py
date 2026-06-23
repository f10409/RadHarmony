"""MONAI dataset for the RSNA 2022 Cervical Spine Fracture Detection Challenge.

2,019 cervical-spine CT volumes (one per study), each loaded from its
per-study DICOM directory by MONAI's ``ITKReader``.  8 study-level
binary labels per volume (``patient_overall`` plus per-vertebra
``c1``..``c7``).  87 of the 2,019 studies ship with a NIfTI
segmentation mask under ``segmentations/<StudyUID>.nii``; passing
``output_mask=True`` filters the dataset to that subset.

For per-slice 2-D bounding-box work see
:class:`RSNA2022CervicalSpineBboxDataset`.
"""

import os
import warnings

import torch
import pandas as pd

from radharmony.harmonizer import RSNA2022CervicalSpineHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform3D


@register_dataset("rsna_2022_cervical_spine")
class RSNA2022CervicalSpineDataset(BaseRadiologicalDataset):
    """3-D MONAI PersistentDataset for RSNA 2022 Cervical Spine.

    Args:
        base_image_dir: Root of the DICOM tree — typically
            ``.../rsna-2022-cervical-spine-fracture-detection/train_images/``.
        csv_path: Path to ``train.csv``.  Auto-inferred (parent of
            ``base_image_dir``) when ``None``.
        segmentation_dir: Directory of NIfTI segmentation masks
            (``<StudyUID>.nii``).  Auto-inferred as ``<parent>/segmentations``
            when ``None``; required when ``output_mask=True`` (otherwise the
            mask filter would silently drop everything).
        transform: MONAI Compose transform.  Defaults to the standard 3-D
            pipeline (ITKReader directory load → HU window → percentile scale
            → resize/pad).
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        hu_window: ``(min_HU, max_HU)`` clipping window applied before
            percentile normalisation.  Default ``(-200, 1800)`` is
            bone-centric (cervical spine target tissue); pass ``(-1000, 1000)``
            for general soft-tissue work.
        output_cls: Yield the 8-D label vector under key ``cls``.
        output_mask: When ``True``, the dataset is restricted to the 87
            studies that have a matching ``<StudyUID>.nii`` segmentation
            (the framework's ``dropna`` on ``mask_path`` handles the filter).
        output_report: Not supported; silently ignored with a warning.
        output_bbox: Not supported; for slice-level bboxes use
            :class:`RSNA2022CervicalSpineBboxDataset` instead.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask"})
    LABEL_COLS = [
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "patient_overall",
    ]
    _HARMONIZER_CLS = RSNA2022CervicalSpineHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        segmentation_dir: str = None,
        transform=None,
        cache_dir: str = "./cache",
        hu_window: tuple[float, float] | None = (-200, 1800),
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
        if segmentation_dir:
            segmentation_dir = os.path.expanduser(segmentation_dir)

        if harmonizer_path is not None and base_image_dir is None:
            _h = RSNA2022CervicalSpineHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_image_dir", None)
            segmentation_dir = segmentation_dir or getattr(_h, "segmentation_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        for flag_name, flag_val in (
            ("output_report", output_report),
            ("output_bbox", output_bbox),
        ):
            if flag_val:
                warnings.warn(
                    f"{flag_name}=True is not supported by RSNA2022CervicalSpineDataset; "
                    "ignoring.  (For per-slice bboxes use RSNA2022CervicalSpineBboxDataset.)",
                    stacklevel=2,
                )
        output_report = False
        output_bbox = False

        # Auto-discover CSV and segmentations dir relative to base_image_dir.
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = infer_path(
                base_image_dir, "train.csv", user_path=csv_path or "",
            ) or csv_path
            if segmentation_dir is None and base_image_dir:
                candidate = os.path.join(os.path.dirname(base_image_dir.rstrip("/\\")), "segmentations")
                if os.path.isdir(candidate):
                    segmentation_dir = candidate
            self._segmentation_dir = segmentation_dir
        else:
            self._csv_path = csv_path
            self._segmentation_dir = segmentation_dir

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_mask:
                output_keys.add("mask")
            transform = RadiologyTransform3D(
                img_size=112, output_keys=output_keys, hu_window=hu_window,
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
        harmonizer = RSNA2022CervicalSpineHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            segmentation_dir=self._segmentation_dir,
        )
        return harmonizer.harmonize()

    def verify_images(self, drop_missing: bool = True) -> pd.DataFrame:
        """Check that every ``image_path`` **directory** (per-study DICOM dir) exists."""
        df = self._get_harmonized_df()

        def _exists(p):
            fp = os.path.join(self.base_image_dir, p) if self.base_image_dir else p
            return os.path.isdir(fp)

        from tqdm import tqdm

        tqdm.pandas(desc="Verifying study dirs")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing study dirs "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All study directories exist.")
        else:
            print(f"Found {len(missing_df)} rows with missing study dirs (not dropped).")

        return missing_df
