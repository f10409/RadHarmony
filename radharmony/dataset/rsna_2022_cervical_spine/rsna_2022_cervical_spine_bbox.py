"""MONAI dataset for the RSNA 2022 Cervical Spine bounding-box subset.

Per-volume 3-D variant of :class:`RSNA2022CervicalSpineDataset`.  Only the
235 studies that have at least one fracture-localizing bounding box.  Each
sample is a full CT volume (the study's DICOM directory).

Bbox format: list of ``[d_min, d_max, y_min, y_max, x_min, x_max]`` 6-tuples
normalised to ``[0, 1]`` in the post-transpose ``(D, H, W)`` axis order.
Multiple fracture annotations within one study produce multiple boxes.

For whole-volume 3-D classification + optional segmentation without bbox,
see :class:`RSNA2022CervicalSpineDataset`.
"""

import os
import warnings

import torch
import pandas as pd

from radharmony.harmonizer import RSNA2022CervicalSpineBboxHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform3D


@register_dataset("rsna_2022_cervical_spine_bbox")
class RSNA2022CervicalSpineBboxDataset(BaseRadiologicalDataset):
    """3-D MONAI PersistentDataset for the cervical-spine bounding-box subset.

    Args:
        base_image_dir: Root of the DICOM tree — typically
            ``.../rsna-2022-cervical-spine-fracture-detection/train_images/``.
        csv_path: Path to ``train.csv`` (the per-study cls labels).
            Auto-inferred when ``None``.
        bbox_csv_path: Path to ``train_bounding_boxes.csv``.
            Auto-inferred when ``None``.
        transform: MONAI Compose transform.  Defaults to the standard 3-D
            pipeline (ITKReader series load → HU window → percentile scale →
            resize/pad).
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls: Yield the 8-D study-level label vector under key ``cls``.
        output_bbox: Yield the per-volume bounding-box list under key ``bbox``.
        output_mask: Not supported; silently ignored.
        output_report: Not supported; silently ignored.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
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
    _HARMONIZER_CLS = RSNA2022CervicalSpineBboxHarmonizer

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
            _h = RSNA2022CervicalSpineBboxHarmonizer.load_from_saved(harmonizer_path)
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

        for flag_name, flag_val in (
            ("output_mask", output_mask),
            ("output_report", output_report),
        ):
            if flag_val:
                warnings.warn(
                    f"{flag_name}=True is not supported by RSNA2022CervicalSpineBboxDataset; ignoring.",
                    stacklevel=2,
                )
        output_mask = False
        output_report = False

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = infer_path(
                base_image_dir, "train.csv", user_path=csv_path or "",
            ) or csv_path
            self._bbox_csv_path = infer_path(
                base_image_dir, "train_bounding_boxes.csv",
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
            transform = RadiologyTransform3D(
                output_keys=output_keys,
                hu_window=(-1000, 1000),
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
        harmonizer = RSNA2022CervicalSpineBboxHarmonizer(
            csv_path=self._csv_path,
            bbox_csv_path=self._bbox_csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()

    def verify_images(self, drop_missing: bool = True) -> pd.DataFrame:
        """Check that every ``image_path`` **directory** (DICOM study dir) exists.

        Overrides the base implementation because ``image_path`` points at a
        study directory, not a single file.
        """
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
            print(
                f"Found {len(missing_df)} rows with missing study dirs (not dropped)."
            )

        return missing_df
