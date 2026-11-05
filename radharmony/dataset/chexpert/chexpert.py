"""MONAI dataset for the CheXpert dataset."""

import torch

from radharmony.harmonizer import CheXpertHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("chexpert")
class CheXpertDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for CheXpert.

    Args:
        base_image_dir: Root directory of the JPEG image tree
            (for example ``CheXpert-v1.0/train/``).
        csv_path: Path to the CheXpert metadata CSV (for example ``train.csv``).
            When ``None``, ``train.csv`` / ``valid.csv`` are searched near
            ``base_image_dir`` using :func:`~radharmony.utils.infer.infer_path`.
        transform: MONAI transform pipeline. Defaults to the standard 2-D pipeline.
        cache_dir: Root directory for MONAI PersistentDataset cache.
        output_cls: Yield label vector under key ``cls``.
        drop_uncertain: If ``True`` (default), drop the rows with an uncertain label (``-1``).
            With ``False``, ``-1`` stays in ``cls`` (D-05).
        not_mentioned: ``"negative"`` (default) gives 0 for a blank label, ``"nan"`` keeps NaN (D-05).
        output_mask / output_report / output_bbox / output_reg: Not supported. A warning,
            and no key (D-11).
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})
    _HARMONIZER_CLS = CheXpertHarmonizer

    LABEL_COLS = [
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

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        drop_uncertain: bool = True,
        not_mentioned: str = "negative",
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
        output_mask: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
    ):
        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_image_dir = _h.base_image_dir

        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        self._drop_uncertain = drop_uncertain
        self._not_mentioned = not_mentioned
        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            csv_path = infer_path(
                base_image_dir,
                "train.csv",
                user_path=csv_path or "",
                required=True,
                arg="csv_path",
            ) or csv_path

        if transform is None:
            transform = RadiologyTransform2D(
                output_keys=output_keys,
                dtype=dtype,
            ).get_transform()
        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,  # not in SUPPORTED_OUTPUTS: the base class warns
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = CheXpertHarmonizer(csv_path=csv_path, base_image_dir=base_image_dir)

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        # Only reached when the class built the harmonizer.
        return self._harmonizer.harmonize(drop_uncertain=self._drop_uncertain, not_mentioned=self._not_mentioned)
