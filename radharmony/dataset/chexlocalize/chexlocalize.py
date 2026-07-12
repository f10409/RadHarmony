"""MONAI dataset for the CheXlocalize dataset."""

import torch

import pandas as pd

from radharmony.harmonizer import CheXlocalizeHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("chexlocalize")
class CheXlocalizeDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for CheXlocalize (CheXpert ``test`` split).

    Args:
        base_image_dir: Root of the CheXpert test image tree
            (e.g. ``.../chexlocalize/CheXpert/test/``).
        csv_path: Path to ``test_labels.csv``. Auto-inferred when ``None``.
        mask_json_path: Path to CheXlocalize's ``gt_segmentations_test.json``
            (COCO-RLE, per-pathology masks). Auto-inferred when ``None``.
            Required (together with ``mask_output_dir``) for ``output_mask=True``.
        mask_output_dir: Directory to write decoded, per-image unioned binary
            mask PNGs. Required when ``output_mask=True``.
        mask_num_cores: Parallel worker threads for mask decoding.
        transform: MONAI transform pipeline. Defaults to the standard 2-D pipeline.
        cache_dir: Root directory for MONAI PersistentDataset cache.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Yield the unioned segmentation mask under key ``mask``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask"})
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
    _HARMONIZER_CLS = CheXlocalizeHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        mask_json_path: str = None,
        mask_output_dir: str = None,
        mask_num_cores: int = 1,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if harmonizer_path is not None and base_image_dir is None:
            _h = CheXlocalizeHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = _h.base_image_dir

        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError("base_image_dir is required when harmonizer_path is not provided.")

        if output_mask and mask_output_dir is None and harmonizer_path is None \
                and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "output_mask=True requires mask_output_dir to be set. CheXlocalize masks "
                "are per-pathology COCO-RLE entries in a JSON file and must be decoded "
                "(and unioned into one mask per image) before MONAI can load them. Pass "
                "mask_json_path and mask_output_dir, e.g.:\n"
                "    CheXlocalizeDataset(..., output_mask=True, "
                "mask_json_path='/path/to/gt_segmentations_test.json', "
                "mask_output_dir='/path/to/masks')"
            )

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = infer_path(
                base_image_dir, "test_labels.csv", user_path=csv_path or "",
            ) or csv_path
            self._mask_json_path = infer_path(
                base_image_dir, "gt_segmentations_test.json", user_path=mask_json_path or "",
            ) or mask_json_path
        else:
            self._csv_path = csv_path
            self._mask_json_path = mask_json_path

        self._mask_output_dir = mask_output_dir
        self._mask_num_cores = mask_num_cores

        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")
        if output_mask:
            output_keys.add("mask")

        if transform is None:
            transform = RadiologyTransform2D(
                output_keys=output_keys, dtype=dtype,
            ).get_transform()

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = CheXlocalizeHarmonizer(
            csv_path=self._csv_path,
            mask_json_path=self._mask_json_path if self.output_mask else None,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize(
            mask_output_dir=self._mask_output_dir,
            mask_num_cores=self._mask_num_cores,
        )
