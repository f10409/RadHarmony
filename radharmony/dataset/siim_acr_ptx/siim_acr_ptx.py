"""MONAI dataset for the SIIM-ACR Pneumothorax dataset."""

import torch

from radharmony.harmonizer import SIIMACRPTXHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("siim_acr_ptx")
class SIIMACRPTXDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for SIIM-ACR Pneumothorax.

    Args:
        base_image_dir: Root directory of the DICOM tree (used for both globbing
            and prepending to relative image paths, e.g. ``…/dicom-images-train/``).
            ``csv_path`` is inferred from the parent directory when not provided
            and the expected file exists.
        csv_path: Path to ``train-rle.csv``.  Defaults to ``<parent>/train-rle.csv``.
        transform: MONAI transform pipeline.  Defaults to the standard 2-D pipeline.
        cache_dir: Root directory for MONAI PersistentDataset cache.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask"})
    LABEL_COLS = ["pneumothorax"]
    _HARMONIZER_CLS = SIIMACRPTXHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        mask_output_dir: str = None,
        mask_num_cores: int = 1,
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if harmonizer_path is not None and base_image_dir is None:
            _h = SIIMACRPTXHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = _h.dicom_dir

        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            csv_path = infer_path(
                base_image_dir, "train-rle.csv",
                user_path=csv_path or "",
            ) or csv_path

        if output_mask and mask_output_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "output_mask=True requires mask_output_dir to be set. "
                "SIIM-ACR PTX masks are RLE strings in the CSV and must be decoded "
                "to image files before MONAI can load them. Pass a directory:\n"
                "    SIIMACRPTXDataset(..., output_mask=True, mask_output_dir='/path/to/masks')"
            )

        output_keys = {"img"}
        if output_cls:    output_keys.add("cls")
        if output_mask:   output_keys.add("mask")
        if output_report: output_keys.add("report")
        if output_bbox:   output_keys.add("bbox")
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
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = SIIMACRPTXHarmonizer(
                csv_path=csv_path,
                dicom_dir=base_image_dir,
            )
        self._mask_output_dir = mask_output_dir
        self._mask_num_cores = mask_num_cores

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize(  # only reached when harmonizer was constructed
            mask_output_dir=self._mask_output_dir,
            mask_num_cores=self._mask_num_cores,
        )
