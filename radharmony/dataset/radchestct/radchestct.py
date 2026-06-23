"""MONAI dataset for the RAD-ChestCT dataset."""

import pandas as pd
import torch

from radharmony.harmonizer import RadChestCTHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform3D


@register_dataset("radchestct")
class RadChestCTDataset(BaseRadiologicalDataset):
    """PyTorch dataset for a single RAD-ChestCT split.

    Args:
        base_image_dir: Root directory containing the ``.npz`` image files.
        csv_path: Path to the metadata CSV (``CT_Scan_Metadata_Complete_35747.csv``).
            Auto-inferred near ``base_image_dir`` when ``None``.
        label_csv_path: Path to one raw findings CSV
            (e.g. ``imgtrain_Abnormality_and_Location_Labels.csv``).
            Auto-inferred near ``base_image_dir`` when ``None``.
        bbox_csv_path: Path to the extrema CSV (``Extrema_35747.csv``). Optional.
            Auto-inferred near ``base_image_dir`` when ``None``.
        transform: MONAI Compose transform. Defaults to a standard 3-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Yield segmentation mask under key ``mask``.
        output_report: Yield report text under key ``report``.
        output_bbox: Yield bounding box under key ``bbox``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask", "report", "bbox"})
    LABEL_COLS = [
        "air_trapping",
        "airspace_disease",
        "aneurysm",
        "arthritis",
        "aspiration",
        "atelectasis",
        "atherosclerosis",
        "bandlike_or_linear",
        "breast_implant",
        "breast_surgery",
        "bronchial_wall_thickening",
        "bronchiectasis",
        "bronchiolectasis",
        "bronchiolitis",
        "bronchitis",
        "cabg",
        "calcification",
        "cancer",
        "cardiomegaly",
        "catheter_or_port",
        "cavitation",
        "chest_tube",
        "clip",
        "congestion",
        "consolidation",
        "coronary_artery_disease",
        "cyst",
        "debris",
        "deformity",
        "density",
        "dilation_or_ectasia",
        "distention",
        "emphysema",
        "fibrosis",
        "fracture",
        "gi_tube",
        "granuloma",
        "groundglass",
        "hardware",
        "heart_failure",
        "heart_valve_replacement",
        "hemothorax",
        "hernia",
        "honeycombing",
        "infection",
        "infiltrate",
        "inflammation",
        "interstitial_lung_disease",
        "lesion",
        "lucency",
        "lung_resection",
        "lymphadenopathy",
        "mass",
        "mucous_plugging",
        "nodule",
        "nodulegr1cm",
        "opacity",
        "other_path",
        "pacemaker_or_defib",
        "pericardial_effusion",
        "pericardial_thickening",
        "plaque",
        "pleural_effusion",
        "pleural_thickening",
        "pneumonia",
        "pneumonitis",
        "pneumothorax",
        "postsurgical",
        "pulmonary_edema",
        "reticulation",
        "scarring",
        "scattered_calc",
        "scattered_nod",
        "secretion",
        "septal_thickening",
        "soft_tissue",
        "staple",
        "stent",
        "sternotomy",
        "suture",
        "tracheal_tube",
        "transplant",
        "tree_in_bud",
        "tuberculosis",
    ]
    _HARMONIZER_CLS = RadChestCTHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_csv_path: str = None,
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
            _h = RadChestCTHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = _h.image_base_dir

        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = infer_path(
                base_image_dir,
                "CT_Scan_Metadata_Complete_35747.csv",
                user_path=csv_path or "",
            ) or csv_path
            self._label_csv_path = infer_path(
                base_image_dir,
                "imgtrain_Abnormality_and_Location_Labels.csv",
                "imgvalid_Abnormality_and_Location_Labels.csv",
                "imgtest_Abnormality_and_Location_Labels.csv",
                user_path=label_csv_path or "",
            ) or label_csv_path
            self._bbox_csv_path = infer_path(
                base_image_dir,
                "Extrema_35747.csv",
                user_path=bbox_csv_path or "",
            ) or bbox_csv_path
        else:
            self._csv_path = csv_path
            self._label_csv_path = label_csv_path
            self._bbox_csv_path = bbox_csv_path

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_bbox:
                output_keys.add("bbox")
                output_keys.add("bbox_labels")
            transform = (
                RadiologyTransform3D(
                    img_size=112,
                    output_keys=output_keys,
                    hu_window=(-1000, 400),
                    base_transpose=False,
                    dtype=dtype,
                )
                .get_transform()
            )

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
        harmonizer = RadChestCTHarmonizer(
            csv_path=self._csv_path,
            label_csv_path=self._label_csv_path,
            image_base_dir=self.base_image_dir,
            bbox_csv_path=self._bbox_csv_path,
        )
        df = harmonizer.harmonize()
        return df
