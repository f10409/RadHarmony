"""MONAI dataset for the 2017 RSNA Pediatric Bone Age Challenge."""

import os
import warnings

import torch

from radharmony.harmonizer import RSNABoneAgeHarmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("rsna_bone_age")
class RSNABoneAgeDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for the RSNA Pediatric Bone Age release.

    Skeletal-age regression on 2-D hand radiographs (PNG).  The official
    training split has 12,611 images; the validation split has 1,425
    spread across two subfolders inside the val ZIP.  Enable
    ``output_reg=True`` to yield ``{"img": ..., "reg": tensor([age_months])}``
    samples.

    Train and val are separate downloads — each takes its own
    ``base_image_dir``.  Use :class:`RSNABoneAgeTrainDataset` or
    :class:`RSNABoneAgeValDataset` to fix the split at construction time.

    Classification and bbox outputs are not supported (the dataset has
    neither); those flags are silently ignored with a warning.

    Args:
        base_image_dir: Split-specific image directory.  For
            ``split="train"`` point at ``boneage-training-dataset/`` (the
            flat dir of ``<id>.png`` files).  For ``split="val"`` point at
            ``Bone Age Validation Set/`` (contains
            ``Validation Dataset.csv`` plus
            ``boneage-validation-dataset-{1,2}/``).
        split: ``"train"`` (default) or ``"val"``.
        csv_path: Optional explicit override for the split's CSV path.
        transform: MONAI Compose transform.  Defaults to the 2-D pipeline.
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls: Not supported; ignored with a warning.
        output_reg: Yield the 1-D regression vector under key ``reg``
            (skeletal age in months).
        output_mask / output_report / output_bbox: Not supported; ignored.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"reg"})
    LABEL_COLS: list = []
    REG_COLS = ["age_months"]
    _HARMONIZER_CLS = RSNABoneAgeHarmonizer

    SPLITS = RSNABoneAgeHarmonizer.SPLITS

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        split: str = "train",
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        output_reg: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if csv_path:
            csv_path = os.path.expanduser(csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = RSNABoneAgeHarmonizer.load_from_saved(harmonizer_path)
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
            ("output_cls", output_cls),
            ("output_mask", output_mask),
            ("output_report", output_report),
            ("output_bbox", output_bbox),
        ):
            if flag_val:
                warnings.warn(
                    f"{flag_name}=True is not supported by RSNABoneAgeDataset "
                    "(regression-only dataset); ignoring.",
                    stacklevel=2,
                )
        output_cls = False
        output_mask = False
        output_report = False
        output_bbox = False

        if split not in self.SPLITS:
            raise ValueError(
                f"split={split!r} is not one of {self.SPLITS}."
            )
        self._split = split

        if transform is None:
            output_keys = {"img"}
            if output_reg:
                output_keys.add("reg")
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
            output_reg=output_reg,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = RSNABoneAgeHarmonizer(
                base_image_dir=base_image_dir,
                csv_path=csv_path,
                split=split,
            )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()
