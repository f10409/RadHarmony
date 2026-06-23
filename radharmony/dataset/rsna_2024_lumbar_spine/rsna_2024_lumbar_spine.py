"""MONAI dataset for the RSNA 2024 Lumbar Spine Degenerative Classification."""

import os
import warnings

import torch

from radharmony.harmonizer import RSNA2024LumbarSpineHarmonizer
from radharmony.harmonizer.rsna_2024_lumbar_spine import _expanded_label_cols
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform3D


@register_dataset("rsna_2024_lumbar_spine")
class RSNA2024LumbarSpineDataset(BaseRadiologicalDataset):
    """3-D MONAI PersistentDataset for RSNA 2024 Lumbar Spine Degenerative Classification.

    One sample per MRI series — 1975 studies × ~3 series = ~6294 rows.  Each
    ``image_path`` is a DICOM series directory (``<study_id>/<series_id>/``)
    that the ITKReader loads as a 3-D volume.  Labels are the 75 one-hot
    severity columns produced by :class:`RSNA2024LumbarSpineHarmonizer`.

    Args:
        base_image_dir: Root DICOM tree, typically
            ``<kagglehub_cache>/competitions/rsna-2024-lumbar-spine-degenerative-classification/train_images/``.
        csv_path: Path to ``train.csv``.  Auto-inferred from the parent of
            ``base_image_dir`` when None.
        series_description_csv_path: Path to ``train_series_descriptions.csv``.
            Auto-inferred when None.
        coord_csv_path: Path to ``train_label_coordinates.csv`` (optional).
            Auto-inferred when None.  When present, enables ``output_bbox``:
            each annotated point becomes a tiny 3-D cube bbox that renders
            as a dot in the bbox overlay pipeline.
        transform: MONAI Compose transform.  Defaults to the 3-D pipeline with
            ``hu_window=None`` (MRI, no Hounsfield window) — percentile-based
            intensity scaling in :func:`_base_load_3d` handles normalisation.
        cache_dir: PersistentDataset cache directory.  None disables caching.
        output_cls: Yield the 75-D label vector under key ``cls``.
        output_mask: Not supported; silently ignored.
        output_report: Not supported; silently ignored.
        output_bbox: Yield per-point cube bboxes under key ``bbox`` (requires
            ``coord_csv_path``).
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = _expanded_label_cols()
    _HARMONIZER_CLS = RSNA2024LumbarSpineHarmonizer

    #: Allowed values for ``series_filter`` — the ``view_position`` strings set
    #: by :class:`RSNA2024LumbarSpineHarmonizer` from ``series_description``.
    SERIES_TYPES = ("Axial T2", "Sagittal T1", "Sagittal T2/STIR")

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        series_description_csv_path: str = None,
        coord_csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        series_filter: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if csv_path:
            csv_path = os.path.expanduser(csv_path)
        if series_description_csv_path:
            series_description_csv_path = os.path.expanduser(series_description_csv_path)
        if coord_csv_path:
            coord_csv_path = os.path.expanduser(coord_csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = RSNA2024LumbarSpineHarmonizer.load_from_saved(harmonizer_path)
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

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            csv_path = infer_path(
                base_image_dir, "train.csv", user_path=csv_path or "",
            ) or csv_path
            series_description_csv_path = infer_path(
                base_image_dir,
                "train_series_descriptions.csv",
                user_path=series_description_csv_path or "",
            ) or series_description_csv_path
            coord_csv_path = infer_path(
                base_image_dir,
                "train_label_coordinates.csv",
                user_path=coord_csv_path or "",
            ) or coord_csv_path

        if output_bbox and not coord_csv_path:
            warnings.warn(
                "output_bbox=True requires train_label_coordinates.csv "
                "(coord_csv_path) — none found. bboxes will be empty.",
                stacklevel=2,
            )

        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")
        if output_bbox:
            output_keys.add("bbox")

        if transform is None:
            transform = RadiologyTransform3D(
                output_keys=output_keys, hu_window=None,
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
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = RSNA2024LumbarSpineHarmonizer(
                csv_path=csv_path,
                series_description_csv_path=series_description_csv_path,
                coord_csv_path=coord_csv_path,
                base_image_dir=base_image_dir,
            )

        if series_filter and series_filter not in self.SERIES_TYPES:
            raise ValueError(
                f"series_filter={series_filter!r} is not one of "
                f"{self.SERIES_TYPES}; pass None to keep all series."
            )
        self._series_filter = series_filter or None

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        df = preset if preset is not None else self._harmonizer.harmonize()
        if self._series_filter:
            df = df[df["view_position"] == self._series_filter].reset_index(drop=True)
        return df

    def verify_images(self, drop_missing: bool = True):
        """Check each series directory exists and contains at least one .dcm.

        Overrides the base implementation because ``image_path`` here is a
        DICOM **series directory** (ITKReader reads the stack), not a single file.
        """
        import pandas as pd
        from tqdm import tqdm

        df = self._get_harmonized_df()

        def _exists(p):
            fp = os.path.join(self.base_image_dir, p) if self.base_image_dir else p
            if not os.path.isdir(fp):
                return False
            try:
                for name in os.listdir(fp):
                    if name.endswith(".dcm"):
                        return True
            except OSError:
                return False
            return False

        tqdm.pandas(desc="Verifying series directories")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing/empty series "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All series directories exist and contain DICOM files.")
        else:
            print(
                f"Found {len(missing_df)} rows with missing/empty series "
                "(not dropped)."
            )
        return missing_df
