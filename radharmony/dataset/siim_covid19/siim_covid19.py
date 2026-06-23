"""MONAI dataset for the 2021 SIIM-FISABIO-RSNA COVID-19 Detection Challenge."""

import os
import warnings

import torch

from radharmony.harmonizer import SIIMCOVID19Harmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("siim_covid19")
class SIIMCOVID19Dataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for the SIIM COVID-19 Detection Challenge.

    ~6,334 chest X-rays (DICOM) across 6,054 studies.  Study-level 4-class
    appearance label (Negative / Typical / Indeterminate / Atypical) with
    optional per-image bounding boxes for airspace opacities.

    Args:
        base_image_dir: Root of the DICOM tree — typically
            ``<kagglehub_cache>/competitions/siim-covid19-detection/train/``.
        csv_path: Path to ``train_study_level.csv``.  Auto-inferred when None.
        image_csv_path: Path to ``train_image_level.csv``.  Auto-inferred when
            None.
        transform: MONAI Compose transform. Defaults to the 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_cls: Yield the 4-D label vector under key ``cls``.
        output_mask: Not supported; silently ignored with a warning.
        output_report: Not supported; silently ignored with a warning.
        output_bbox: Yield bounding boxes under key ``bbox``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = [
        "atypical_appearance",
        "indeterminate_appearance",
        "negative_for_pneumonia",
        "typical_appearance",
    ]
    _HARMONIZER_CLS = SIIMCOVID19Harmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        image_csv_path: str = None,
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
        if image_csv_path:
            image_csv_path = os.path.expanduser(image_csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = SIIMCOVID19Harmonizer.load_from_saved(harmonizer_path)
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
                    f"{flag_name}=True is not supported by SIIMCOVID19Dataset; ignoring.",
                    stacklevel=2,
                )
        output_mask = False
        output_report = False

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            csv_path = infer_path(
                base_image_dir, "train_study_level.csv", user_path=csv_path or "",
            ) or csv_path
            image_csv_path = infer_path(
                base_image_dir,
                "train_image_level.csv",
                user_path=image_csv_path or "",
            ) or image_csv_path

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
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = SIIMCOVID19Harmonizer(
                csv_path=csv_path,
                image_csv_path=image_csv_path,
                base_image_dir=base_image_dir,
            )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()
