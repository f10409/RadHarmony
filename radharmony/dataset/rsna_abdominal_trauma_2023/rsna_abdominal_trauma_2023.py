"""MONAI dataset for the RSNA 2023 Abdominal Trauma Detection Challenge.

4,711 abdominal CT volumes (one per series, ~1.5 series per patient on
average), each loaded from its DICOM series directory by MONAI's
``ITKReader``.  14 patient-level binary labels per volume covering
bowel/extravasation injury (binary) and kidney/liver/spleen severity
(3-class one-hot per organ), replicated across each patient's series.
"""

import os
import warnings

import torch
import pandas as pd

from radharmony.harmonizer import RSNAAbdominalTrauma2023Harmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform3D


@register_dataset("rsna_abdominal_trauma_2023")
class RSNAAbdominalTrauma2023Dataset(BaseRadiologicalDataset):
    """3-D MONAI PersistentDataset for RSNA 2023 Abdominal Trauma Detection.

    Args:
        base_image_dir: Root of the DICOM tree — typically
            ``.../rsna-2023-abdominal-trauma-detection/train_images/``.
        csv_path: Path to ``train_2024.csv`` (patient-level labels).
            Auto-inferred via ``infer_path`` from ``base_image_dir`` when
            ``None``.
        series_meta_csv_path: Path to ``train_series_meta.csv`` (series-level
            metadata).  Auto-inferred when ``None``.
        transform: MONAI Compose transform.  Defaults to the standard 3-D
            pipeline (ITKReader series load → HU window → percentile scale →
            resize/pad).
        cache_dir: PersistentDataset cache directory.  ``None`` disables.
        hu_window: ``(min_HU, max_HU)`` clipping window applied before
            percentile normalisation.  Default ``(-150, 250)`` covers the
            soft-tissue / contrast-enhanced abdomen — different from CTPA.
        output_cls: Yield the 14-D label vector under key ``cls``.
        output_mask: Not supported in this initial integration; silently
            ignored with a warning.  See the dataset's ``segmentations/``
            directory for organ masks if you need them.
        output_report: Not supported (no reports in this release).
        output_bbox: Not supported in this initial integration; the
            ``image_level_labels_2024.csv`` Active_Extravasation point
            annotations could be wired as bbox dots in a follow-up.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})
    LABEL_COLS = [
        "any_injury",
        "bowel_healthy",
        "bowel_injury",
        "extravasation_healthy",
        "extravasation_injury",
        "kidney_healthy",
        "kidney_high",
        "kidney_low",
        "liver_healthy",
        "liver_high",
        "liver_low",
        "spleen_healthy",
        "spleen_high",
        "spleen_low",
    ]
    _HARMONIZER_CLS = RSNAAbdominalTrauma2023Harmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        series_meta_csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        hu_window: tuple[float, float] | None = (-150, 250),
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
        if series_meta_csv_path:
            series_meta_csv_path = os.path.expanduser(series_meta_csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = RSNAAbdominalTrauma2023Harmonizer.load_from_saved(harmonizer_path)
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
            ("output_bbox", output_bbox),
        ):
            if flag_val:
                warnings.warn(
                    f"{flag_name}=True is not supported by "
                    "RSNAAbdominalTrauma2023Dataset; ignoring.",
                    stacklevel=2,
                )
        output_mask = False
        output_report = False
        output_bbox = False

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = (
                infer_path(base_image_dir, "train_2024.csv", user_path=csv_path or "")
                or csv_path
            )
            self._series_meta_csv_path = (
                infer_path(
                    base_image_dir,
                    "train_series_meta.csv",
                    user_path=series_meta_csv_path or "",
                )
                or series_meta_csv_path
            )
        else:
            self._csv_path = csv_path
            self._series_meta_csv_path = series_meta_csv_path

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            transform = RadiologyTransform3D(
                img_size=112,
                output_keys=output_keys,
                hu_window=hu_window,
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
        harmonizer = RSNAAbdominalTrauma2023Harmonizer(
            csv_path=self._csv_path,
            series_meta_csv_path=self._series_meta_csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()

    def verify_images(self, drop_missing: bool = True) -> pd.DataFrame:
        """Check that every ``image_path`` **directory** (DICOM series dir) exists.

        Overrides the base implementation because ``image_path`` points at a
        series directory, not a single file.
        """
        df = self._get_harmonized_df()

        def _exists(p):
            fp = (
                os.path.join(self.base_image_dir, p)
                if self.base_image_dir
                else p
            )
            return os.path.isdir(fp)

        from tqdm import tqdm

        tqdm.pandas(desc="Verifying series dirs")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing series dirs "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All series directories exist.")
        else:
            print(
                f"Found {len(missing_df)} rows with missing series dirs "
                "(not dropped)."
            )

        return missing_df
