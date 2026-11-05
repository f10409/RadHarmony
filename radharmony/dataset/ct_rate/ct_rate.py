"""MONAI dataset for the CT-RATE dataset."""

import os

import torch

from radharmony.harmonizer import CTRATEHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform3D


# Label and metadata file names of each split, in the release's "tables/" folder.
# The release names the valid labels "valid_*" but the valid metadata "validation_*".
_TABLES = {
    "train": (("train_predicted_labels.csv",), ("train_metadata.csv",)),
    "valid": (("valid_predicted_labels.csv", "validation_predicted_labels.csv"),
              ("validation_metadata.csv", "valid_metadata.csv")),
}


def _folder_split(base_image_dir: str) -> str:
    """Return "valid" when the image folder name starts with "valid" (for example valid_fixed/), else "train"."""
    name = os.path.basename(os.path.normpath(base_image_dir or ""))
    return "valid" if name.startswith("valid") else "train"


def _in_tables(names: tuple) -> list:
    """Look in the sibling tables/ folder first: it is fast, and it avoids the image trees."""
    return [f"tables/{n}" for n in names] + list(names)


@register_dataset("ct_rate")
class CTRATEDataset(BaseRadiologicalDataset):
    """3-D MONAI PersistentDataset for CT-RATE.

    Args:
        base_image_dir: Root directory of the NIfTI volume tree (for example ``…/dataset/train_fixed/``).
            ``csv_path`` and ``view_position_csv_path`` are inferred from the ``tables/``
            subdirectory of the parent when not provided and the expected files exist.
        csv_path: Path to ``train_predicted_labels.csv``.  Defaults to
            ``<parent>/tables/train_predicted_labels.csv``.
        view_position_csv_path: Path to ``train_metadata.csv``.  Defaults to
            ``<parent>/tables/train_metadata.csv``.
        transform: MONAI transform pipeline.  Defaults to the standard 3-D pipeline.
        cache_dir: Root directory for MONAI PersistentDataset cache.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})
    _HARMONIZER_CLS = CTRATEHarmonizer
    # None: the image folder name picks the split of the tables (train or valid).
    _SPLIT = None

    LABEL_COLS = [
        "arterial_wall_calcification",
        "atelectasis",
        "bronchiectasis",
        "cardiomegaly",
        "consolidation",
        "coronary_artery_wall_calcification",
        "emphysema",
        "hiatal_hernia",
        "interlobular_septal_thickening",
        "lung_nodule",
        "lung_opacity",
        "lymphadenopathy",
        "medical_material",
        "mosaic_attenuation_pattern",
        "peribronchial_thickening",
        "pericardial_effusion",
        "pleural_effusion",
        "pulmonary_fibrotic_sequela",
    ]

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        view_position_csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        hu_window: tuple[float, float] | None = (-1000, 1000),
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_image_dir = _h.base_image_dir

        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            label_names, meta_names = _TABLES[self._SPLIT or _folder_split(base_image_dir)]
            csv_path = infer_path(
                base_image_dir, *_in_tables(label_names), user_path=csv_path or "", required=True, arg="csv_path",
            ) or csv_path
            view_position_csv_path = infer_path(
                base_image_dir, *_in_tables(meta_names), user_path=view_position_csv_path or "",
            ) or view_position_csv_path

        # Only cls is supported. The base class warns for the other flags. A key
        # that no sample has must not be in the transform: it raises KeyError (D-11).
        output_keys = {"img"}
        if output_cls:    output_keys.add("cls")
        if transform is None:
            transform = RadiologyTransform3D(
                output_keys=output_keys, hu_window=hu_window,
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
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = self._HARMONIZER_CLS(
                csv_path=csv_path,
                view_position_csv_path=view_position_csv_path,
                base_image_dir=base_image_dir,
            )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()  # only reached when harmonizer was constructed
