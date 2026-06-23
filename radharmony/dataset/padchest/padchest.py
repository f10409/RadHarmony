"""MONAI dataset for the PadChest (BIMCV) dataset."""

import pandas as pd
import torch

from radharmony.harmonizer import PadChestHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("padchest")
class PadChestDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for PadChest (BIMCV).

    160,861 frontal + lateral chest X-rays (PNG) from Hospital San Juan de
    Alicante, Spain. Labels (193 distinct findings) are NLP-derived from
    Spanish reports plus manual review. Spanish radiology reports are
    available via ``output_report=True``.

    Images are distributed across 52 sibling subdirectories under
    ``base_image_dir`` (``0/``, ``1/``, …, ``50/``, plus ``54/``;
    slots 51/52/53 don't exist in the official distribution).

    Args:
        base_image_dir: Root directory containing ``0/``, ``1/``, …,
            ``54/`` subdirectories (e.g. ``/mnt/.../PadChest/images/``).
        csv_path: Path to
            ``PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz``.
            Auto-inferred from ``base_image_dir``'s ancestors when ``None``.
        transform: MONAI Compose transform. Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_cls: Yield 193-element label vector under key ``cls``.
        output_report: Yield Spanish report text under key ``report``.
        drop_uncertain: PadChest has no CheXpert-style ``-1`` uncertainty,
            but the annotators flagged ~3,151 images with the meta-labels
            ``exclude`` or ``suboptimal_study``.  When ``True`` (default),
            those rows are dropped during harmonization.  Pass ``False`` to
            keep them — useful when training on the full noisy distribution.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "report"})
    LABEL_COLS = [
        "abnormal_foreign_body",
        "abscess",
        "adenopathy",
        "air_bronchogram",
        "air_fluid_level",
        "air_trapping",
        "alveolar_pattern",
        "aortic_aneurysm",
        "aortic_atheromatosis",
        "aortic_button_enlargement",
        "aortic_elongation",
        "aortic_endoprosthesis",
        "apical_pleural_thickening",
        "artificial_aortic_heart_valve",
        "artificial_heart_valve",
        "artificial_mitral_heart_valve",
        "asbestosis_signs",
        "ascendent_aortic_elongation",
        "atelectasis",
        "atelectasis_basal",
        "atypical_pneumonia",
        "axial_hyperostosis",
        "azygoesophageal_recess_shift",
        "azygos_lobe",
        "blastic_bone_lesion",
        "bone_cement",
        "bone_metastasis",
        "breast_mass",
        "bronchiectasis",
        "bronchovascular_markings",
        "bullas",
        "calcified_adenopathy",
        "calcified_densities",
        "calcified_fibroadenoma",
        "calcified_granuloma",
        "calcified_mediastinal_adenopathy",
        "calcified_pleural_plaques",
        "calcified_pleural_thickening",
        "callus_rib_fracture",
        "cardiomegaly",
        "catheter",
        "cavitation",
        "central_vascular_redistribution",
        "central_venous_catheter",
        "central_venous_catheter_via_jugular_vein",
        "central_venous_catheter_via_subclavian_vein",
        "central_venous_catheter_via_umbilical_vein",
        "cervical_rib",
        "chest_drain_tube",
        "chilaiditi_sign",
        "chronic_changes",
        "clavicle_fracture",
        "consolidation",
        "copd_signs",
        "costochondral_junction_hypertrophy",
        "costophrenic_angle_blunting",
        "cyst",
        "dai",
        "descendent_aortic_elongation",
        "dextrocardia",
        "diaphragmatic_eventration",
        "double_j_stent",
        "dual_chamber_device",
        "electrical_device",
        "emphysema",
        "empyema",
        "end_on_vessel",
        "endoprosthesis",
        "endotracheal_tube",
        "esophagic_dilatation",
        "exclude",
        "external_foreign_body",
        "fibrotic_band",
        "fissure_thickening",
        "flattened_diaphragm",
        "fracture",
        "gastrostomy_tube",
        "goiter",
        "granuloma",
        "ground_glass_pattern",
        "gynecomastia",
        "heart_insufficiency",
        "heart_valve_calcified",
        "hemidiaphragm_elevation",
        "hiatal_hernia",
        "hilar_congestion",
        "hilar_enlargement",
        "humeral_fracture",
        "humeral_prosthesis",
        "hydropneumothorax",
        "hyperinflated_lung",
        "hypoexpansion",
        "hypoexpansion_basal",
        "increased_density",
        "infiltrates",
        "interstitial_pattern",
        "kerley_lines",
        "kyphosis",
        "laminar_atelectasis",
        "lepidic_adenocarcinoma",
        "lipomatosis",
        "lobar_atelectasis",
        "loculated_fissural_effusion",
        "loculated_pleural_effusion",
        "lung_metastasis",
        "lung_vascular_paucity",
        "lymphangitis_carcinomatosa",
        "lytic_bone_lesion",
        "major_fissure_thickening",
        "mammary_prosthesis",
        "mass",
        "mastectomy",
        "mediastinal_enlargement",
        "mediastinal_mass",
        "mediastinal_shift",
        "mediastinic_lipomatosis",
        "metal",
        "miliary_opacities",
        "minor_fissure_thickening",
        "multiple_nodules",
        "nephrostomy_tube",
        "nipple_shadow",
        "nodule",
        "non_axial_articular_degenerative_changes",
        "normal",
        "nsg_tube",
        "obesity",
        "osteopenia",
        "osteoporosis",
        "osteosynthesis_material",
        "pacemaker",
        "pectum_carinatum",
        "pectum_excavatum",
        "pericardial_effusion",
        "pleural_effusion",
        "pleural_mass",
        "pleural_plaques",
        "pleural_thickening",
        "pneumomediastinum",
        "pneumonia",
        "pneumoperitoneo",
        "pneumothorax",
        "post_radiotherapy_changes",
        "prosthesis",
        "pseudonodule",
        "pulmonary_artery_enlargement",
        "pulmonary_artery_hypertension",
        "pulmonary_edema",
        "pulmonary_fibrosis",
        "pulmonary_hypertension",
        "pulmonary_mass",
        "pulmonary_venous_hypertension",
        "reservoir_central_venous_catheter",
        "respiratory_distress",
        "reticular_interstitial_pattern",
        "reticulonodular_interstitial_pattern",
        "rib_fracture",
        "right_sided_aortic_arch",
        "round_atelectasis",
        "sclerotic_bone_lesion",
        "scoliosis",
        "segmental_atelectasis",
        "single_chamber_device",
        "soft_tissue_mass",
        "sternoclavicular_junction_hypertrophy",
        "sternotomy",
        "subacromial_space_narrowing",
        "subcutaneous_emphysema",
        "suboptimal_study",
        "superior_mediastinal_enlargement",
        "supra_aortic_elongation",
        "surgery",
        "surgery_breast",
        "surgery_heart",
        "surgery_humeral",
        "surgery_lung",
        "surgery_neck",
        "suture_material",
        "thoracic_cage_deformation",
        "total_atelectasis",
        "tracheal_shift",
        "tracheostomy_tube",
        "tuberculosis",
        "tuberculosis_sequelae",
        "unchanged",
        "vascular_hilar_enlargement",
        "vascular_redistribution",
        "ventriculoperitoneal_drain_tube",
        "vertebral_anterior_compression",
        "vertebral_compression",
        "vertebral_degenerative_changes",
        "vertebral_fracture",
        "volume_loss",
    ]
    _HARMONIZER_CLS = PadChestHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
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
        drop_uncertain: bool = True,
    ):
        self._drop_uncertain = drop_uncertain
        if harmonizer_path is not None and base_image_dir is None:
            _h = PadChestHarmonizer.load_from_saved(harmonizer_path)
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
            self._csv_path = (
                infer_path(
                    base_image_dir,
                    "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
                    "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
                    user_path=csv_path or "",
                )
                or csv_path
            )
        else:
            self._csv_path = csv_path

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_report:
                output_keys.add("report")
            transform = RadiologyTransform2D(
                img_size=224,
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

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = PadChestHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize(drop_uncertain=self._drop_uncertain)
