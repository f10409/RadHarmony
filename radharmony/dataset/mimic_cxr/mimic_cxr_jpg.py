"""MONAI dataset for the MIMIC-CXR-JPG dataset (pre-converted JPEG images)."""

import torch

from radharmony.harmonizer import MIMICCXRJPGHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("mimic_cxr_jpg")
class MIMICCXRJPGDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for MIMIC-CXR-JPG, all splits in one table.

    This class mixes the official train, validate and test splits. Do not use
    it for a benchmark. Use :class:`MIMICCXRJPGTrainDataset`,
    :class:`MIMICCXRJPGValidateDataset` and :class:`MIMICCXRJPGTestDataset`.

    Args:
        base_image_dir: Root directory of the JPEG image tree (for example ``…/2.0.0/files/``).
        csv_path: Path to ``mimic-cxr-2.0.0-metadata.csv``.  Defaults to
            ``mimic-cxr-2.0.0-metadata.csv`` in the parent of ``base_image_dir``.
        label_csv_path: Path to ``mimic-cxr-2.0.0-chexpert.csv``.  Defaults to
            ``mimic-cxr-2.0.0-chexpert.csv`` in the parent of ``base_image_dir``.
            With ``output_cls=True``, a missing file raises ``FileNotFoundError`` (D-35).
        transform: MONAI transform pipeline.  Defaults to the standard 2-D pipeline.
        cache_dir: Root directory for MONAI PersistentDataset cache.
        drop_uncertain: If ``True`` (default), drop the rows with an uncertain label (``-1``).
            With ``False``, ``-1`` stays in ``cls`` (D-05).
        not_mentioned: ``"negative"`` (default) gives 0 for a blank label, ``"nan"`` keeps NaN (D-05).
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})
    _HARMONIZER_CLS = MIMICCXRJPGHarmonizer

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
        label_csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
        drop_uncertain: bool = True,
        not_mentioned: str = "negative",
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        self._drop_uncertain = drop_uncertain
        self._not_mentioned = not_mentioned
        # When a saved harmonizer is provided, infer base_image_dir from its
        # stored base_image_dir so the caller need not repeat it.
        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_image_dir = _h.base_image_dir

        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        # Only search for CSVs when not loading from a saved harmonizer.
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            csv_path = infer_path(
                base_image_dir,
                "mimic-cxr-2.0.0-metadata.csv", "mimic-cxr-2.0.0-metadata.csv.gz",
                user_path=csv_path or "",
                required=True,
                arg="csv_path",
            ) or csv_path
            # output_cls needs the label file. Without it, a sample has no cls (D-35).
            label_csv_path = infer_path(
                base_image_dir,
                "mimic-cxr-2.0.0-chexpert.csv", "mimic-cxr-2.0.0-chexpert.csv.gz",
                user_path=label_csv_path or "",
                required=output_cls,
                arg="label_csv_path",
            ) or label_csv_path

        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")
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
            output_mask=output_mask,
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )
        # Build the harmonizer only when it is used.
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = MIMICCXRJPGHarmonizer(
                csv_path=csv_path,
                label_csv_path=label_csv_path,
                base_image_dir=base_image_dir,
            )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        # Only reached when the class built the harmonizer.
        return self._harmonizer.harmonize(drop_uncertain=self._drop_uncertain, not_mentioned=self._not_mentioned)
