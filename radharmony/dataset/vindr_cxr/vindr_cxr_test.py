"""MONAI dataset for the VinDr-CXR test dataset."""

import pandas as pd
import torch

from radharmony.harmonizer import VinDrCXRTestHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("vindr_cxr_test")
class VinDrCXRTestDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for VinDr-CXR test set.

    All images are PA-view DICOM chest X-rays stored as
    ``<image_id>.dicom`` under ``base_image_dir``
    (e.g. ``.../vindr-cxr/1.0.0/test/``).

    Labels and bounding boxes are already aggregated (verified by 5
    radiologists).  Unlike the training set, no majority-vote step is
    needed.

    Note: the test CSV uses ``"Other disease"`` (singular); the harmonizer
    renames it to ``"Other diseases"`` to match the train set convention.

    Args:
        base_image_dir: Root directory containing ``.dicom`` files
            (e.g. ``.../vindr-cxr/1.0.0/test/``).
        csv_path: Path to ``image_labels_test.csv``.  Auto-inferred
            near ``base_image_dir`` when ``None``.
        bbox_csv_path: Path to ``annotations_test.csv``.  Auto-inferred
            near ``base_image_dir`` when ``None``.
        transform: MONAI Compose transform. Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_bbox: Yield bounding-box list under key ``bbox``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = [
        "aortic_enlargement",
        "atelectasis",
        "calcification",
        "cardiomegaly",
        "clavicle_fracture",
        "consolidation",
        "copd",
        "edema",
        "emphysema",
        "enlarged_pa",
        "ild",
        "infiltration",
        "lung_cavity",
        "lung_cyst",
        "lung_opacity",
        "lung_tumor",
        "mediastinal_shift",
        "no_finding",
        "nodule/mass",
        "other_diseases",
        "other_lesion",
        "pleural_effusion",
        "pleural_thickening",
        "pneumonia",
        "pneumothorax",
        "pulmonary_fibrosis",
        "rib_fracture",
        "tuberculosis",
    ]
    _HARMONIZER_CLS = VinDrCXRTestHarmonizer

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
        if harmonizer_path is not None and base_image_dir is None:
            _h = VinDrCXRTestHarmonizer.load_from_saved(harmonizer_path)
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
            self._csv_path = infer_path(
                base_image_dir,
                "image_labels_test.csv",
                user_path=csv_path or "",
            ) or csv_path
            self._bbox_csv_path = infer_path(
                base_image_dir,
                "annotations_test.csv",
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
                output_keys.add("bbox_labels")
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

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = VinDrCXRTestHarmonizer(
            csv_path=self._csv_path,
            bbox_csv_path=self._bbox_csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
