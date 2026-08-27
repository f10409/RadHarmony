"""Gradio app for visualising RadHarmony dataset outputs and transform effects.

Launch with:
    python app.py
or:
    gradio app.py

Adding a new dataset
--------------------
1. Write a build function ``_build_<name>(base_dir, csv_path, extra_field,
   extra_field2, cache_dir, **output_flags) -> (dataset_obj, error_str | None)``.
2. Add one entry to ``DATASET_REGISTRY``.
That is all — the UI and loader update automatically.
"""

import importlib
import os
import random
from dataclasses import dataclass
from typing import Callable

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from radharmony.dataset import (
    BRAXDataset,
    BRAXPNGDataset,
    ChestXray14Dataset,
    ChestXray14BboxDataset,
    ChestXray14TrainDataset,
    ChestXray14TestDataset,
    CheXpertDataset,
    CheXpertTrainDataset,
    CheXpertValidDataset,
    CheXpertPlusDataset,
    CheXlocalizeDataset,
    MIMICCXRDataset,
    MIMICCXRJPGDataset,
    MIMICCXRJPGTestDataset,
    PadChestDataset,
    RadChestCTDataset,
    ReXGradientTrainDataset,
    ReXGradientValidDataset,
    ReXGradientTestDataset,
    RANZCRClipDataset,
    RSNA2022CervicalSpineDataset,
    RSNA2022CervicalSpineBboxDataset,
    RSNA2022CervicalSpineTrainDataset,
    RSNA2022CervicalSpineTestDataset,
    RSNA2024LumbarSpineDataset,
    RSNA2024LumbarSpineTrainDataset,
    RSNA2024LumbarSpineTestDataset,
    RSNAAbdominalTrauma2023Dataset,
    RSNAAbdominalTrauma2023TrainDataset,
    RSNAAbdominalTrauma2023TestDataset,
    RSNABoneAgeDataset,
    RSNABoneAgeTrainDataset,
    RSNABoneAgeValDataset,
    RSNAPEDetectionDataset,
    RSNAPEDetectionTrainDataset,
    RSNAPEDetectionTestDataset,
    RSNAPneumoniaDataset,
    RSNAPneumoniaKaggleDataset,
    RSNAPneumoniaKaggleTrainDataset,
    RSNAPneumoniaKaggleTestDataset,
    SIIMACRPTXDataset,
    SIIMACRPTXTrainDataset,
    SIIMACRPTXTestDataset,
    SIIMCOVID19Dataset,
    SIIMCOVID19TrainDataset,
    SIIMCOVID19TestDataset,
    CTRATEDataset,
    TAIXRay512Dataset,
    TAIXRayDataset,
    MontgomeryCXRDataset,
    OpenICXRDataset,
    ShenzhenCXRDataset,
    VinDrCXRTestDataset,
    VinDrCXRTrainDataset,
    VinDrPCXRDataset,
    EmoryCXRDataset,
    MSCXRDataset,
    MSCXRTDataset,
    ChestImaGenomeGoldDataset,
    ChestImaGenomeSilverDataset,
)
from radharmony.dataset.transforms import RadiologyTransform2D, RadiologyTransform3D
from radharmony.utils.data_utils import get_data_dict
from radharmony.utils.infer import infer_path as _infer


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


#: Controlled vocabulary of imaging modalities.  Add new values here first so
#: the UI grouping and dataset registry stay in sync.  Kept short on purpose —
#: the tab strip gets crowded fast if we inflate this list.
SUPPORTED_MODALITIES: tuple = (
    "CXR",  # chest X-ray
    "Radiograph",  # general radiography: hand / bone / extremity / etc. (anything not chest)
    "CT",  # computed tomography
    "MRI",  # magnetic resonance imaging
    "VQA",  # vision-question-answer datasets (any underlying imaging type)
    "Other",
)


@dataclass
class DatasetConfig:
    """Describes one dataset entry in the visualizer.

    Attributes:
        is_3d: Whether to use the 3-D transform pipeline and HU controls.
        extra_field_label: UI label for the first dataset-specific textbox.
        build: Callable that instantiates the dataset. Signature::

            build(base_dir, csv_path, extra_field, extra_field2, cache_dir, **output_flags)
                -> (dataset_obj, error_str | None)

            Return ``(None, "message")`` to abort with a user-facing error.
        modality: Imaging modality (see :data:`SUPPORTED_MODALITIES`) used only
            for UI grouping — independent of ``is_3d`` which drives the
            transform pipeline.  Mostly correlated with dimensionality (CXR → 2-D,
            CT → 3-D) but not identical: some MRI acquisitions are 2-D, etc.
        extra_field2_label: UI label for the optional second textbox (empty = hidden).
    """

    is_3d: bool
    extra_field_label: str
    build: Callable
    modality: str
    extra_field2_label: str = ""
    csv_label: str = "CSV path (optional)"
    base_dir_placeholder: str = ""
    csv_placeholder: str = ""
    extra_placeholder: str = ""
    extra_placeholder2: str = ""
    # Optional dataset-specific dropdown (e.g. series-type filter).
    # Empty label hides the control.  Choices must include a "(all)" sentinel
    # if the dataset wants to support "no filter".
    extra_dropdown_label: str = ""
    extra_dropdown_choices: tuple = ()
    extra_dropdown_kwarg: str = ""  # kwarg name forwarded to the build function
    # Group-header sentinel: appears in dropdown as a visual separator;
    # selecting it is a no-op (load_dataset returns early).
    is_group_header: bool = False

    def __post_init__(self) -> None:
        if self.is_group_header:
            return
        if self.modality not in SUPPORTED_MODALITIES:
            raise ValueError(
                f"modality={self.modality!r} is not in SUPPORTED_MODALITIES "
                f"({', '.join(SUPPORTED_MODALITIES)}).  Extend the tuple if "
                "this dataset introduces a genuinely new modality."
            )


_APP_DTYPE = torch.float32  # float32 works on CPU and all GPUs


def _build_brax(cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv = _infer(
        base_dir,
        "master_spreadsheet.csv",
        "master_spreadsheet_update.csv",
        user_path=csv_path,
    )
    if not csv:
        return None, "master_spreadsheet.csv is required."
    strategy = (extra_field or "raw").strip()
    return (
        cls(
            base_image_dir=base_dir,
            csv_path=csv,
            uncertain_strategy=strategy,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_brax_dicom(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_brax(BRAXDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_brax_png(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_brax(BRAXPNGDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_chexpert_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv = _infer(
        base_dir, "train.csv", "CheXpert-v1.0-small/train.csv", user_path=csv_path
    )
    return (
        CheXpertTrainDataset(
            base_image_dir=base_dir,
            csv_path=csv or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chexpert_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv = _infer(
        base_dir, "valid.csv", "CheXpert-v1.0-small/valid.csv", user_path=csv_path
    )
    return (
        CheXpertValidDataset(
            base_image_dir=base_dir,
            csv_path=csv or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chexpert_plus(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv = _infer(
        base_dir,
        "df_chexpert_plus_240401.csv",
        user_path=csv_path,
    )
    label_json = _infer(
        base_dir,
        "report_fixed.json",
        user_path=extra_field,
    )
    return (
        CheXpertPlusDataset(
            base_image_dir=base_dir,
            csv_path=csv or None,
            label_json_path=label_json or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_report=flags.get("output_report", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chexlocalize(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    csv = _infer(base_dir, "test_labels.csv", user_path=csv_path)
    if not csv:
        return None, "test_labels.csv is required."
    mask_json = _infer(
        base_dir, "gt_segmentations_test.json", user_path=extra_field
    ) or None
    mask_dir = extra_field2 or None
    if flags.get("output_mask") and mask_dir is None:
        return (
            None,
            "Mask output dir is required when Mask is enabled for CheXlocalize.",
        )
    return (
        CheXlocalizeDataset(
            base_image_dir=base_dir,
            csv_path=csv,
            mask_json_path=mask_json,
            mask_output_dir=mask_dir,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_mimic(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    csv = _infer(
        base_dir, "cxr-record-list.csv.gz", "cxr-record-list.csv", user_path=csv_path
    )
    if not csv:
        return (
            None,
            "Metadata CSV (cxr-record-list.csv.gz) not found. Please provide the path explicitly.",
        )
    label = _infer(
        base_dir,
        "mimic-cxr-2.0.0-chexpert.csv",
        "mimic-cxr-2.0.0-chexpert.csv.gz",
        user_path=extra_field,
    )
    report = _infer(
        base_dir, "cxr-study-list.csv.gz", "cxr-study-list.csv", user_path=extra_field2
    )
    return (
        MIMICCXRDataset(
            base_image_dir=base_dir,
            csv_path=csv,
            label_csv_path=label or None,
            report_csv_path=report or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=flags.get("output_report", False) and bool(report),
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_mimic_jpg(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    csv = _infer(
        base_dir,
        "mimic-cxr-2.0.0-metadata.csv.gz",
        "mimic-cxr-2.0.0-metadata.csv",
        user_path=csv_path,
    )
    label = _infer(
        base_dir,
        "mimic-cxr-2.0.0-chexpert.csv",
        "mimic-cxr-2.0.0-chexpert.csv.gz",
        user_path=extra_field,
    )
    return (
        MIMICCXRJPGDataset(
            base_image_dir=base_dir,
            csv_path=csv or None,
            label_csv_path=label or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=False,
        ),
        None,
    )


def _build_mimic_jpg_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv = _infer(
        base_dir,
        "mimic-cxr-2.0.0-metadata.csv.gz",
        "mimic-cxr-2.0.0-metadata.csv",
        user_path=csv_path,
    )
    label = _infer(
        base_dir,
        "mimic-cxr-2.1.0-test-set-labeled.csv",
        user_path=extra_field,
    )
    return (
        MIMICCXRJPGTestDataset(
            base_image_dir=base_dir,
            csv_path=csv or None,
            label_csv_path=label or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
        ),
        None,
    )


def _build_ctrate(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    csv = _infer(
        base_dir,
        "train_predicted_labels.csv",
        # CT-RATE uses both naming variants for the validation split
        # (PhysioNet ships labels as valid_*.csv, metadata as validation_*.csv).
        "valid_predicted_labels.csv",
        "validation_predicted_labels.csv",
        user_path=csv_path,
    )
    view_pos = _infer(
        base_dir,
        "train_metadata.csv",
        "validation_metadata.csv",
        "valid_metadata.csv",
        user_path=extra_field,
    )
    return (
        CTRATEDataset(
            base_image_dir=base_dir,
            csv_path=csv or None,
            view_position_csv_path=view_pos or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_radchestct(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv_path = _infer(
        base_dir, "CT_Scan_Metadata_Complete_35747.csv", user_path=csv_path
    )
    if not csv_path:
        return None, "Metadata CSV (CT_Scan_Metadata_Complete_35747.csv) is required."
    extra_field = _infer(
        base_dir,
        "imgtrain_Abnormality_and_Location_Labels.csv",
        "imgvalid_Abnormality_and_Location_Labels.csv",
        "imgtest_Abnormality_and_Location_Labels.csv",
        user_path=extra_field,
    )
    if not extra_field:
        return (
            None,
            "Label CSV (imgtrain/imgvalid/imgtest_Abnormality_and_Location_Labels.csv) is required.",
        )
    bbox_csv = (
        _infer(
            base_dir,
            "Extrema_35747.csv",
            user_path=extra_field2,
        )
        or None
    )
    return (
        RadChestCTDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            label_csv_path=extra_field,
            bbox_csv_path=bbox_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chestxray14_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv_path = _infer(base_dir, "Data_Entry_2017.csv", user_path=csv_path)
    if not csv_path:
        return None, "Data_Entry_2017.csv is required."
    bbox_csv = _infer(base_dir, "BBox_List_2017.csv", user_path=extra_field) or None
    return (
        ChestXray14TrainDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            bbox_csv_path=bbox_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chestxray14_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv_path = _infer(base_dir, "Data_Entry_2017.csv", user_path=csv_path)
    if not csv_path:
        return None, "Data_Entry_2017.csv is required."
    bbox_csv = _infer(base_dir, "BBox_List_2017.csv", user_path=extra_field) or None
    return (
        ChestXray14TestDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            bbox_csv_path=bbox_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chestxray14_bbox(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv_path = _infer(base_dir, "Data_Entry_2017.csv", user_path=csv_path)
    if not csv_path:
        return None, "Data_Entry_2017.csv is required."
    bbox_csv = _infer(base_dir, "BBox_List_2017.csv", user_path=extra_field)
    if not bbox_csv:
        return None, "BBox_List_2017.csv is required for the bbox subset."
    return (
        ChestXray14BboxDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            bbox_csv_path=bbox_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_padchest(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv_path = _infer(
        base_dir,
        "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
        "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv",
        user_path=csv_path,
    )
    if not csv_path:
        return None, "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv(.gz) is required."
    return (
        PadChestDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=flags.get("output_report", False),
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rexgradient(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, split, dataset_cls, json_name, **flags
):
    """Common build path for the three ReXGradient splits.

    The JSON metadata is the source of truth; the per-study CSV that
    ships alongside cannot produce per-image rows. Passing a CSV path
    is rejected with an explicit error so users don't silently get the
    wrong data shape.
    """
    csv_path = _infer(base_dir, json_name, user_path=csv_path)
    if not csv_path:
        return None, f"{json_name} is required (per-image JSON metadata)."
    if str(csv_path).lower().endswith(".csv"):
        return (
            None,
            "ReXGradient requires the per-image JSON metadata "
            f"({json_name}); the per-study CSV does not contain image paths.",
        )
    return (
        dataset_cls(
            base_image_dir=base_dir,
            csv_path=csv_path,
            cache_dir=cache_dir or None,
            output_cls=False,
            output_mask=False,
            output_report=flags.get("output_report", False),
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rexgradient_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_rexgradient(
        base_dir, csv_path, extra_field, extra_field2, cache_dir,
        split="train",
        dataset_cls=ReXGradientTrainDataset,
        json_name="train_metadata_view_position.json",
        **flags,
    )


def _build_rexgradient_valid(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_rexgradient(
        base_dir, csv_path, extra_field, extra_field2, cache_dir,
        split="valid",
        dataset_cls=ReXGradientValidDataset,
        json_name="valid_metadata_view_position.json",
        **flags,
    )


def _build_rexgradient_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_rexgradient(
        base_dir, csv_path, extra_field, extra_field2, cache_dir,
        split="test",
        dataset_cls=ReXGradientTestDataset,
        json_name="test_metadata_view_position.json",
        **flags,
    )


def _build_vindr_cxr_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv_path = _infer(
        base_dir,
        "image_labels_train.csv",
        user_path=csv_path,
    )
    if not csv_path:
        return None, "Label CSV (image_labels_train.csv) is required."
    bbox_csv = (
        _infer(
            base_dir,
            "annotations_train.csv",
            user_path=extra_field,
        )
        or None
    )
    return (
        VinDrCXRTrainDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            bbox_csv_path=bbox_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chest_imagenome_gold(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    # base_dir = MIMIC-CXR DICOM files/ root; extra_field = Chest ImaGenome root.
    annotation_dir = extra_field or None
    if not annotation_dir:
        return None, "Annotation dir (Chest ImaGenome release root) is required."
    return (
        ChestImaGenomeGoldDataset(
            base_image_dir=base_dir,
            annotation_dir=annotation_dir,
            cache_dir=cache_dir or None,
            output_bbox=flags.get("output_bbox", True),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chest_imagenome_silver(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    # base_dir = MIMIC-CXR DICOM files/ root; extra_field = Chest ImaGenome root.
    annotation_dir = extra_field or None
    if not annotation_dir:
        return None, "Annotation dir (Chest ImaGenome release root) is required."
    split = flags.get("split") or "all"
    return (
        ChestImaGenomeSilverDataset(
            base_image_dir=base_dir,
            annotation_dir=annotation_dir,
            split=split,
            cache_dir=cache_dir or None,
            output_bbox=flags.get("output_bbox", True),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_vindr_cxr_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv_path = _infer(
        base_dir,
        "image_labels_test.csv",
        user_path=csv_path,
    )
    if not csv_path:
        return None, "Label CSV (image_labels_test.csv) is required."
    bbox_csv = (
        _infer(
            base_dir,
            "annotations_test.csv",
            user_path=extra_field,
        )
        or None
    )
    return (
        VinDrCXRTestDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            bbox_csv_path=bbox_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_vindr_pcxr(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, "Dataset root directory (containing train/ and test/) is required."
    return (
        VinDrPCXRDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_bbox=flags.get("output_bbox", False),
            output_mask=False,
            output_report=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_2024_lumbar_spine_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    extra_field = os.path.expanduser(extra_field) if extra_field else extra_field

    resolved_csv = _infer(base_dir, "train.csv", user_path=csv_path)
    if not resolved_csv:
        return None, "train.csv is required."
    resolved_series = _infer(
        base_dir,
        "train_series_descriptions.csv",
        user_path=extra_field,
    )
    if not resolved_series:
        return None, "train_series_descriptions.csv is required."
    return (
        RSNA2024LumbarSpineTrainDataset(
            base_image_dir=base_dir,
            csv_path=resolved_csv,
            series_description_csv_path=resolved_series,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_bbox=flags.get("output_bbox", False),
            series_filter=flags.get("series_filter"),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_2024_lumbar_spine_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    return (
        RSNA2024LumbarSpineTestDataset(
            base_image_dir=base_dir,
            series_description_csv_path=csv_path or None,
            cache_dir=cache_dir or None,
            series_filter=flags.get("series_filter"),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_siim_covid19_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    extra_field = os.path.expanduser(extra_field) if extra_field else extra_field

    study_csv = _infer(base_dir, "train_study_level.csv", user_path=csv_path)
    if not study_csv:
        return None, "train_study_level.csv is required."
    image_csv = _infer(base_dir, "train_image_level.csv", user_path=extra_field)
    if not image_csv:
        return None, "train_image_level.csv is required."
    return (
        SIIMCOVID19TrainDataset(
            base_image_dir=base_dir,
            csv_path=study_csv,
            image_csv_path=image_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_siim_covid19_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    return (
        SIIMCOVID19TestDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_siim_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    mask_dir = extra_field or None
    if flags.get("output_mask") and mask_dir is None:
        return (
            None,
            "Mask output dir is required when Mask is enabled for SIIM-ACR-PTX.",
        )
    return (
        SIIMACRPTXTrainDataset(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            cache_dir=cache_dir or None,
            mask_output_dir=mask_dir,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_siim_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    return (
        SIIMACRPTXTestDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_pneumonia_kaggle_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    class_csv = _infer(base_dir, "stage_2_detailed_class_info.csv", user_path=csv_path)
    if not class_csv:
        return None, "stage_2_detailed_class_info.csv is required."
    bbox_csv = (
        _infer(base_dir, "stage_2_train_labels.csv", user_path=extra_field) or None
    )
    return (
        RSNAPneumoniaKaggleTrainDataset(
            base_image_dir=base_dir,
            csv_path=class_csv,
            bbox_csv_path=bbox_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_pneumonia_kaggle_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    return (
        RSNAPneumoniaKaggleTestDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_bone_age_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    return (
        RSNABoneAgeTrainDataset(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            cache_dir=cache_dir or None,
            output_reg=flags.get("output_reg", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_bone_age_val(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    return (
        RSNABoneAgeValDataset(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            cache_dir=cache_dir or None,
            output_reg=flags.get("output_reg", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_pe_detection_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path

    resolved_csv = _infer(base_dir, "train.csv", user_path=csv_path)
    if not resolved_csv:
        return None, "train.csv is required."
    return (
        RSNAPEDetectionTrainDataset(
            base_image_dir=base_dir,
            csv_path=resolved_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_pe_detection_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    return (
        RSNAPEDetectionTestDataset(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            cache_dir=cache_dir or None,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_abdominal_trauma_2023_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    extra_field = os.path.expanduser(extra_field) if extra_field else extra_field

    resolved_csv = _infer(base_dir, "train_2024.csv", user_path=csv_path)
    if not resolved_csv:
        return None, "train_2024.csv is required."
    resolved_series = _infer(base_dir, "train_series_meta.csv", user_path=extra_field)
    if not resolved_series:
        return None, "train_series_meta.csv is required."
    return (
        RSNAAbdominalTrauma2023TrainDataset(
            base_image_dir=base_dir,
            csv_path=resolved_csv,
            series_meta_csv_path=resolved_series,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
        ),
        None,
    )


def _build_rsna_abdominal_trauma_2023_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    return (
        RSNAAbdominalTrauma2023TestDataset(
            base_image_dir=base_dir,
            series_meta_csv_path=csv_path or None,
            cache_dir=cache_dir or None,
        ),
        None,
    )


def _build_rsna_pneumonia(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    # Expand ~ in both paths — Gradio passes the raw string without shell expansion
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path

    json_path = _infer(
        base_dir,
        "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json",
        user_path=csv_path,
    )
    if not json_path:
        return None, "Adjudicated JSON annotation file is required."
    return (
        RSNAPneumoniaDataset(
            base_image_dir=base_dir,
            csv_path=json_path,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rsna_2022_cervical_spine_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    csv = _infer(base_dir, "train.csv", user_path=csv_path)
    seg_dir = os.path.expanduser(extra_field) if extra_field else None
    return (
        RSNA2022CervicalSpineTrainDataset(
            base_image_dir=base_dir,
            csv_path=csv or None,
            segmentation_dir=seg_dir,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
        ),
        None,
    )


def _build_rsna_2022_cervical_spine_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    return (
        RSNA2022CervicalSpineTestDataset(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            cache_dir=cache_dir or None,
        ),
        None,
    )


def _build_rsna_2022_cervical_spine_bbox(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    bbox_path = os.path.expanduser(extra_field) if extra_field else None
    csv = _infer(base_dir, "train.csv", user_path=csv_path)
    bbox_csv = _infer(base_dir, "train_bounding_boxes.csv", user_path=bbox_path or "")
    if not bbox_csv:
        return None, "train_bounding_boxes.csv is required for the bbox subset."
    return (
        RSNA2022CervicalSpineBboxDataset(
            base_image_dir=base_dir,
            csv_path=csv or None,
            bbox_csv_path=bbox_csv,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_taix_ray(
    cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv = _infer(base_dir, "annotation.csv", user_path=csv_path)
    return (
        cls(
            base_image_dir=base_dir,
            csv_path=csv or None,
            cache_dir=cache_dir or None,
            label_mode=flags.get("label_mode", "binary"),
            output_cls=flags.get("output_cls", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_taix_ray_512(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return _build_taix_ray(
        TAIXRay512Dataset,
        base_dir,
        csv_path,
        extra_field,
        extra_field2,
        cache_dir,
        **flags,
    )


def _build_taix_ray_original(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return _build_taix_ray(
        TAIXRayDataset,
        base_dir,
        csv_path,
        extra_field,
        extra_field2,
        cache_dir,
        **flags,
    )


def _build_openi_cxr(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, "Dataset root directory (containing ecgen-radiology/ and images/) is required."
    return (
        OpenICXRDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_report=flags.get("output_report", False),
            output_mask=False,
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_openi_cxr_dicom(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    """OpenI variant that loads DICOM files instead of PNG."""
    import pandas as pd
    from radharmony.harmonizer.openi_cxr import OpenICXRHarmonizer

    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, "Dataset root directory (containing ecgen-radiology/ and DICOM folders) is required."

    try:
        h = OpenICXRHarmonizer(base_dir=base_dir)
        df = h.harmonize()
    except Exception as e:
        return None, f"Harmonization error: {e}"

    missing_dcm = df["dicom_path"].isna().sum()
    df = df.dropna(subset=["dicom_path"]).copy()
    if df.empty:
        return None, "No DICOM paths found. Ensure NLMCXR_dcm.tgz has been extracted."

    # Swap image_path to point at DICOM files
    df["image_path"] = df["dicom_path"]

    return (
        OpenICXRDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_report=flags.get("output_report", False),
            output_mask=False,
            output_bbox=False,
            harmonized_df=df,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_montgomery_cxr(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, "Dataset root directory (containing CXR_png/) is required."
    mask_dir = os.path.expanduser(extra_field) if extra_field else None
    if flags.get("output_mask") and mask_dir is None:
        return (
            None,
            "Mask output dir is required when Mask is enabled for Montgomery CXR "
            "(left+right lung PNGs are fused into a single binary PNG per image "
            "and written here before MONAI can load them).",
        )
    return (
        MontgomeryCXRDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            mask_output_dir=mask_dir,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_shenzhen_cxr(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, "Dataset root directory (containing CXR_png/) is required."
    mask_dir = os.path.expanduser(extra_field) if extra_field else None
    if flags.get("output_mask") and mask_dir is None:
        return (
            None,
            "Mask output dir is required when Mask is enabled for Shenzhen CXR "
            "(per-finding PNGs are unioned into a single 'TB region' mask per "
            "TB-positive image and written here before MONAI can load them).",
        )
    return (
        ShenzhenCXRDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            mask_output_dir=mask_dir,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_emory_cxr(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.harmonizer.emory_cxr import EmoryCXRHarmonizer

    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    meta_csv = (csv_path or "").strip() or None
    # Metadata CSV is required and cannot be auto-discovered — the CSVs live in
    # a separate TABLES/ folder, not in the (potentially millions-of-files) image tree.
    if not meta_csv:
        return None, (
            "CSV path is required for EmoryCXR v2. "
            "Enter the full path to the EmoryCXR v2 metadata CSV."
        )
    label = (extra_field or "").strip() or None
    report = (extra_field2 or "").strip() or None
    h = EmoryCXRHarmonizer(
        csv_path=meta_csv,
        base_image_dir=base_dir,
        label_csv_path=label,
        report_csv_path=report,
    )
    harmonized_df = h.harmonize()
    return (
        EmoryCXRDataset(
            base_image_dir=base_dir,
            harmonized_df=harmonized_df,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False) and label is not None,
            output_mask=False,
            output_report=flags.get("output_report", False) and report is not None,
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_ms_cxr(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    if not base_dir:
        return None, "MIMIC-CXR files/ directory (JPG or DICOM tree) is required."
    if not csv_path:
        return None, "CSV path (MS-CXR local alignment CSV) is required."
    return (
        MSCXRDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_bbox=flags.get("output_bbox", False),
            output_mask=False,
            output_report=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_ms_cxr_t(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    if not base_dir:
        return None, "MIMIC-CXR-JPG root (containing files/) is required."
    if not csv_path:
        return None, "CSV path (MS-CXR-T temporal classification CSV) is required."
    return (
        MSCXRTDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            cache_dir=cache_dir or None,
            output_cls=False,
            output_mask=False,
            output_report=False,
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _header(modality: str) -> DatasetConfig:
    """Create a non-selectable group-header separator for the dropdown."""
    return DatasetConfig(
        False, "", lambda *a, **k: (None, None), modality=modality, is_group_header=True
    )


def _build_ranzcr_clip(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv = _infer(base_dir, "train.csv", user_path=csv_path)
    if not csv:
        return None, "train.csv is required."
    annotations_csv = _infer(
        base_dir, "train_annotations.csv", user_path=extra_field
    ) or None
    mask_dir = extra_field2 or None
    if flags.get("output_mask") and mask_dir is None:
        return (
            None,
            "Mask output dir is required when Mask is enabled for RANZCR-CLIP.",
        )
    return (
        RANZCRClipDataset(
            base_image_dir=base_dir,
            csv_path=csv,
            annotations_csv_path=annotations_csv,
            mask_output_dir=mask_dir,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_roco(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    """ROCO: multimodal radiology captioning dataset from PubMed Central.

    base_dir   → roco-dataset root directory (contains data/)
    extra_field → comma-separated splits to include (default: train,validation,test)
    """
    from radharmony.harmonizer.roco import ROCOHarmonizer
    from tqdm import tqdm

    base_dir_exp = os.path.expanduser(base_dir) if base_dir else None
    if not base_dir_exp or not os.path.isdir(base_dir_exp):
        return None, "ROCO base_dir (roco-dataset root with data/) not found."

    data_dir = os.path.join(base_dir_exp, "data")
    if not os.path.isdir(data_dir):
        return None, f"Expected data/ subdirectory not found in {base_dir_exp}."

    # Parse optional split filter from extra_field
    splits = None
    if extra_field and extra_field.strip():
        splits = [s.strip() for s in extra_field.strip().split(",") if s.strip()]

    try:
        h = ROCOHarmonizer(
            base_dir=base_dir_exp,
            splits=splits,
            radiology_only=True,
        )
        df = h.harmonize()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"

    class _ROCOAdapter:
        LABEL_COLS = []
        REG_COLS   = []
        _cols = {
            "caption": "answer",
            "keywords": "keywords",
        }

        def __init__(self, dataframe):
            self._df = dataframe

        def get_harmonized_df(self):
            return self._df

        def verify_images(self, drop_missing=True):
            def _exists(p):
                fp = os.path.join(base_dir_exp, p) if base_dir_exp else p
                return os.path.isfile(fp)
            tqdm.pandas(desc="Verifying ROCO images")
            mask = self._df["image_path"].progress_apply(_exists)
            missing = self._df[~mask].copy()
            if not missing.empty and drop_missing:
                self._df = self._df[mask].reset_index(drop=True)
            return missing

    return _ROCOAdapter(df), None


def _build_gemex_vqa(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    """GEMeX-VQA: chest X-ray VQA over MIMIC-CXR images.

    base_dir    → MIMIC-CXR-JPG files/ root (images live here)
    extra_field → directory with the 4 GEMeX-VQA JSONL files
    """
    import json as _json
    from radharmony.harmonizer import GEMeXVQAHarmonizer
    from tqdm import tqdm

    data_dir = os.path.expanduser(extra_field) if extra_field else None
    base_dir_exp = os.path.expanduser(base_dir) if base_dir else None

    if not data_dir:
        return None, "GEMeX-VQA data dir (JSONL files) is required in the Extra field."
    if not os.path.isdir(data_dir):
        return None, f"GEMeX-VQA data dir not found: {data_dir}"

    try:
        h = GEMeXVQAHarmonizer(data_dir=data_dir, base_image_dir=base_dir_exp)
        df = h.harmonize()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"

    # --- Build per-image Q&A summary grouped by question subtype ---
    _SUBTYPE_LABELS = {
        "closed_ended":  "Closed Ended (Yes / No)",
        "open_ended":    "Open Ended",
        "single_choice": "Single Choice",
        "multi_choice":  "Multi Choice",
    }

    # Pre-build a mapping: image_path → {subtype → [question dicts]}
    _image_qa: dict = {}
    for _, row in df.iterrows():
        ip = row["image_path"]
        st = row["question_subtype"]
        _image_qa.setdefault(ip, {s: [] for s in _SUBTYPE_LABELS})
        _image_qa[ip][st].append({
            "question": row["question"],
            "choices":  row.get("choices"),
            "answer":   row["answer"],
            "type":     row["question_type"],
        })

    def _format_qa(image_path):
        sections = []
        for subtype, label in _SUBTYPE_LABELS.items():
            qs = _image_qa.get(image_path, {}).get(subtype, [])
            if not qs:
                continue
            sections.append(f"── {label} ──")
            for qi in qs:
                sections.append(f"Q ({qi['type']}): {qi['question']}")
                choices = qi["choices"]
                if choices:
                    answer_letters = {a.strip() for a in qi["answer"].split(",")}
                    fmt = "   ".join(
                        f"{c} ✓" if c.split(":")[0].strip() in answer_letters else c
                        for c in choices
                    )
                    sections.append(f"   {fmt}")
                    sections.append(f"   → {qi['answer']}")
                else:
                    sections.append(f"   → {qi['answer']}")
                sections.append("")
        return "\n".join(sections).strip()

    df["qa_text"] = df["image_path"].apply(_format_qa)

    # --- Bounding boxes: visual_locations are [x1,y1,x2,y2] in 256×256 space.
    # Normalize to [y_min, y_max, x_min, x_max] in [0,1] for the transform pipeline. ---
    _COORD = 256.0

    def _norm_bboxes(row):
        locs = row.get("visual_locations") or []
        bboxes = []
        for loc in locs:
            if len(loc) == 4:
                x1, y1, x2, y2 = loc
                bboxes.append([
                    round(y1 / _COORD, 6),
                    round(y2 / _COORD, 6),
                    round(x1 / _COORD, 6),
                    round(x2 / _COORD, 6),
                ])
        return _json.dumps(bboxes) if bboxes else None

    def _bbox_labels_fn(row):
        regions = row.get("visual_regions") or []
        return _json.dumps(regions) if regions else None

    df["bbox_norm"]       = df.apply(_norm_bboxes, axis=1)
    df["bbox_labels_json"] = df.apply(_bbox_labels_fn, axis=1)

    class _GEMeXAdapter:
        LABEL_COLS = []
        REG_COLS = []
        _cols = {
            "report":      "ori_report",        # radiology report → Report box
            "qa_text":     "qa_text",            # all Q&A for image → Q&A box
            "bbox":        "bbox_norm",
            "bbox_labels": "bbox_labels_json",
        }

        def __init__(self, dataframe):
            self._df = dataframe

        def get_harmonized_df(self):
            return self._df

        def verify_images(self, drop_missing=True):
            def _exists(p):
                fp = os.path.join(base_dir_exp, p) if base_dir_exp else p
                return os.path.isfile(fp)
            tqdm.pandas(desc="Verifying GEMeX-VQA images")
            mask = self._df["image_path"].progress_apply(_exists)
            missing = self._df[~mask].copy()
            if not missing.empty and drop_missing:
                self._df = self._df[mask].reset_index(drop=True)
            return missing

    return _GEMeXAdapter(df), None


# To add a dataset: write a _build_* function above, then add one line here.
DATASET_REGISTRY: dict[str, DatasetConfig] = {
    # ── CXR datasets ─────────────────────────────────────────────────────────
    "── BRAX ──": _header("CXR"),
    "BRAX (DICOM)": DatasetConfig(
        False,
        "Uncertain strategy (raw / u_zeros / u_ones / u_ignore / drop)",
        _build_brax_dicom,
        modality="CXR",
        base_dir_placeholder="e.g. /data/physionet.org/files/brax/1.1.0/",
        csv_placeholder="auto: master_spreadsheet.csv",
        extra_placeholder="raw (default — preserves source 1/0/-1/NaN)",
    ),
    "BRAX (PNG)": DatasetConfig(
        False,
        "Uncertain strategy (raw / u_zeros / u_ones / u_ignore / drop)",
        _build_brax_png,
        modality="CXR",
        base_dir_placeholder="e.g. /data/physionet.org/files/brax/1.1.0/",
        csv_placeholder="auto: master_spreadsheet.csv",
        extra_placeholder="raw (default — preserves source 1/0/-1/NaN)",
    ),
    "── CheXpert ──": _header("CXR"),
    "CheXpert (Train)": DatasetConfig(
        False,
        "",
        _build_chexpert_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/CheXpert-v1.0/train/",
        csv_placeholder="auto: train.csv",
    ),
    "CheXpert (Valid)": DatasetConfig(
        False,
        "",
        _build_chexpert_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/CheXpert-v1.0/valid/",
        csv_placeholder="auto: valid.csv",
    ),
    "CheXpert-Plus": DatasetConfig(
        False,
        "Label JSON (optional)",
        _build_chexpert_plus,
        modality="CXR",
        base_dir_placeholder="e.g. /data/chexpertplus/DICOM/Uncompressed/",
        csv_placeholder="auto: df_chexpert_plus_240401.csv",
        extra_placeholder="auto: report_fixed.json",
    ),
    "CheXlocalize": DatasetConfig(
        False,
        "Segmentation JSON (optional)",
        _build_chexlocalize,
        modality="CXR",
        extra_field2_label="Mask output dir (required for output_mask)",
        base_dir_placeholder="e.g. /data/chexlocalize/CheXpert/test/",
        csv_placeholder="auto: test_labels.csv",
        extra_placeholder="auto: gt_segmentations_test.json",
    ),
    "── ChestX-ray14 ──": _header("CXR"),
    "ChestX-ray14 (Train)": DatasetConfig(
        False,
        "BBox CSV (to exclude bbox images)",
        _build_chestxray14_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/NIH_CXR/CXR14/",
        csv_placeholder="auto: Data_Entry_2017.csv",
        extra_placeholder="auto: BBox_List_2017.csv",
    ),
    "ChestX-ray14 (Test)": DatasetConfig(
        False,
        "BBox CSV (to exclude bbox images)",
        _build_chestxray14_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/NIH_CXR/CXR14/",
        csv_placeholder="auto: Data_Entry_2017.csv",
        extra_placeholder="auto: BBox_List_2017.csv",
    ),
    "ChestX-ray14 (BBox)": DatasetConfig(
        False,
        "BBox CSV",
        _build_chestxray14_bbox,
        modality="CXR",
        base_dir_placeholder="e.g. /data/NIH_CXR/CXR14/",
        csv_placeholder="auto: Data_Entry_2017.csv",
        extra_placeholder="auto: BBox_List_2017.csv",
    ),
    "── PadChest ──": _header("CXR"),
    "PadChest": DatasetConfig(
        False,
        "",
        _build_padchest,
        modality="CXR",
        base_dir_placeholder="e.g. /data/PadChest/images/",
        csv_placeholder="auto: PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
    ),
    "── ReXGradient-160K ──": _header("CXR"),
    "ReXGradient-160K (Train)": DatasetConfig(
        False,
        "",
        _build_rexgradient_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/ReXGradient-160K/deid_png/",
        csv_placeholder="auto: train_metadata_view_position.json",
    ),
    "ReXGradient-160K (Valid)": DatasetConfig(
        False,
        "",
        _build_rexgradient_valid,
        modality="CXR",
        base_dir_placeholder="e.g. /data/ReXGradient-160K/deid_png/",
        csv_placeholder="auto: valid_metadata_view_position.json",
    ),
    "ReXGradient-160K (Test)": DatasetConfig(
        False,
        "",
        _build_rexgradient_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/ReXGradient-160K/deid_png/",
        csv_placeholder="auto: test_metadata_view_position.json",
    ),
    "── MIMIC-CXR ──": _header("CXR"),
    "MIMIC-CXR": DatasetConfig(
        False,
        "Label CSV (optional)",
        _build_mimic,
        modality="CXR",
        extra_field2_label="Report CSV (optional)",
        base_dir_placeholder="e.g. /data/mimic-cxr/2.1.0/files/",
        csv_placeholder="auto: cxr-record-list.csv.gz",
        extra_placeholder="from MIMIC-CXR-JPG: mimic-cxr-2.0.0-chexpert.csv",
        extra_placeholder2="auto: cxr-study-list.csv.gz",
    ),
    "MIMIC-CXR-JPG": DatasetConfig(
        False,
        "Label CSV (optional)",
        _build_mimic_jpg,
        modality="CXR",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
        csv_placeholder="auto: mimic-cxr-2.0.0-metadata.csv.gz",
        extra_placeholder="auto: mimic-cxr-2.0.0-chexpert.csv",
    ),
    "MIMIC-CXR-JPG (Test)": DatasetConfig(
        False,
        "Test labels CSV",
        _build_mimic_jpg_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
        csv_placeholder="auto: mimic-cxr-2.0.0-metadata.csv.gz",
        extra_placeholder="auto: mimic-cxr-2.1.0-test-set-labeled.csv",
    ),
    "── SIIM-ACR Pneumothorax ──": _header("CXR"),
    "SIIM-ACR-PTX (Train)": DatasetConfig(
        False,
        "Mask output dir",
        _build_siim_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/SIIM_ACR_Pneumothorax/dicom-images-train/",
        csv_placeholder="auto: train-rle.csv",
    ),
    "SIIM-ACR-PTX (Test)": DatasetConfig(
        False,
        "",
        _build_siim_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/SIIM_ACR_Pneumothorax/dicom-images-test/",
    ),
    "── RANZCR CLiP ──": _header("CXR"),
    "RANZCR CLiP": DatasetConfig(
        False,
        "Annotations CSV (optional)",
        _build_ranzcr_clip,
        modality="CXR",
        extra_field2_label="Mask output dir (required for output_mask)",
        base_dir_placeholder="e.g. ~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/",
        csv_placeholder="auto: train.csv",
        extra_placeholder="auto: train_annotations.csv",
    ),
    "── SIIM COVID-19 ──": _header("CXR"),
    "SIIM COVID-19 (Train)": DatasetConfig(
        False,
        "Image-level CSV (train_image_level.csv)",
        _build_siim_covid19_train,
        modality="CXR",
        base_dir_placeholder="e.g. ~/datasets/competitions/siim-covid19-detection/train/",
        csv_placeholder="auto: train_study_level.csv",
        extra_placeholder="auto: train_image_level.csv",
    ),
    "SIIM COVID-19 (Test)": DatasetConfig(
        False,
        "",
        _build_siim_covid19_test,
        modality="CXR",
        base_dir_placeholder="e.g. ~/datasets/competitions/siim-covid19-detection/test/",
    ),
    "── TAI X-Ray ──": _header("CXR"),
    "TAIX-Ray (512px)": DatasetConfig(
        False,
        "",
        _build_taix_ray_512,
        modality="CXR",
        base_dir_placeholder="e.g. /data/TAIX-Ray/data_512/images/",
        csv_placeholder="auto: annotation.csv",
        extra_dropdown_label="Label mode",
        extra_dropdown_choices=("binary", "ordinal"),
        extra_dropdown_kwarg="label_mode",
    ),
    "TAIX-Ray (original)": DatasetConfig(
        False,
        "",
        _build_taix_ray_original,
        modality="CXR",
        base_dir_placeholder="e.g. /data/TAIX-Ray/data_original/images/",
        csv_placeholder="auto: annotation.csv",
        extra_dropdown_label="Label mode",
        extra_dropdown_choices=("binary", "ordinal"),
        extra_dropdown_kwarg="label_mode",
    ),
    "── VinDr-CXR ──": _header("CXR"),
    "VinDr-CXR (Train)": DatasetConfig(
        False,
        "BBox CSV (optional)",
        _build_vindr_cxr_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/VinDr-CXR/vindr-cxr/1.0.0/train/",
        csv_placeholder="auto: image_labels_train.csv",
        extra_placeholder="auto: annotations_train.csv",
    ),
    "VinDr-CXR (Test)": DatasetConfig(
        False,
        "BBox CSV (optional)",
        _build_vindr_cxr_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/VinDr-CXR/vindr-cxr/1.0.0/test/",
        csv_placeholder="auto: image_labels_test.csv",
        extra_placeholder="auto: annotations_test.csv",
    ),
    "── VinDr-PCXR ──": _header("CXR"),
    "VinDr-PCXR": DatasetConfig(
        False,
        "",
        _build_vindr_pcxr,
        modality="CXR",
        base_dir_placeholder="e.g. /data/VINDR-PCXR/",
        csv_placeholder="(not required, parsed from image_labels CSV files)",
    ),
    "── OpenI IU CXR ──": _header("CXR"),
    "OpenI IU CXR (PNG)": DatasetConfig(
        False,
        "",
        _build_openi_cxr,
        modality="CXR",
        base_dir_placeholder="e.g. /data/openi-iu-cxr/",
        csv_placeholder="(not required — parsed from ecgen-radiology/ XML)",
    ),
    "OpenI IU CXR (DICOM)": DatasetConfig(
        False,
        "",
        _build_openi_cxr_dicom,
        modality="CXR",
        base_dir_placeholder="e.g. /data/openi-iu-cxr/",
        csv_placeholder="(not required — parsed from ecgen-radiology/ XML)",
    ),
    "── Montgomery CXR ──": _header("CXR"),
    "Montgomery County CXR": DatasetConfig(
        False,
        "Mask output dir (required for output_mask)",
        _build_montgomery_cxr,
        modality="CXR",
        base_dir_placeholder="e.g. /data/MontgomerySet/CXR_png/",
        csv_placeholder="(not required — label is in filename)",
        extra_placeholder="e.g. /data/MontgomerySet/fused_lung_masks/",
    ),
    "── Shenzhen CXR ──": _header("CXR"),
    "Shenzhen Hospital CXR": DatasetConfig(
        False,
        "Mask output dir (required for output_mask)",
        _build_shenzhen_cxr,
        modality="CXR",
        base_dir_placeholder="e.g. /data/Shenzhen-Hospital-CXR-Set/CXR_png/",
        csv_placeholder="(not required — label is in filename)",
        extra_placeholder="e.g. /data/Shenzhen-Hospital-CXR-Set/tb_region_masks/",
    ),
    "── EmoryCXR ──": _header("CXR"),
    "EmoryCXR v2": DatasetConfig(
        False,
        "Label CSV (finding labels)",
        _build_emory_cxr,
        modality="CXR",
        extra_field2_label="Report CSV",
        base_dir_placeholder="e.g. /path/to/EmoryCXRv2/DEID_PNG/",
        csv_placeholder="REQUIRED: full path to the EmoryCXR v2 metadata CSV",
        extra_placeholder="e.g. /path/to/EmoryCXRv2/TABLES/<finding-label CSV>",
        extra_placeholder2="e.g. /path/to/EmoryCXRv2/TABLES/<report CSV>",
    ),
    "── MS-CXR ──": _header("CXR"),
    "MS-CXR (phrase grounding)": DatasetConfig(
        False,
        "",
        _build_ms_cxr,
        modality="CXR",
        csv_label="CSV path (required — MS-CXR local alignment CSV)",
        base_dir_placeholder="MIMIC-CXR files/ root (JPG or DICOM), e.g. /data/MIMIC-CXR-V2-AWS/files/",
        csv_placeholder="e.g. /data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv",
    ),
    "MS-CXR-T (temporal progression)": DatasetConfig(
        False,
        "",
        _build_ms_cxr_t,
        modality="CXR",
        csv_label="CSV path (required — MS-CXR-T temporal classification CSV)",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/",
        csv_placeholder="e.g. /data/ms-cxr-t/MS_CXR_T_temporal_image_classification_v1.0.0.csv",
    ),
    "── Chest ImaGenome ──": _header("CXR"),
    "Chest ImaGenome (Gold)": DatasetConfig(
        False,
        "Annotation dir (Chest ImaGenome root)",
        _build_chest_imagenome_gold,
        modality="CXR",
        base_dir_placeholder="MIMIC-CXR DICOM files/ root, e.g. /data/MIMIC-CXR-V2-AWS/files/",
        csv_label="(not used - annotations come from the annotation dir)",
        extra_placeholder="e.g. /data/CHEST-IMAGENOME/",
    ),
    "Chest ImaGenome (Silver)": DatasetConfig(
        False,
        "Annotation dir (Chest ImaGenome root)",
        _build_chest_imagenome_silver,
        modality="CXR",
        base_dir_placeholder="MIMIC-CXR DICOM files/ root, e.g. /data/MIMIC-CXR-V2-AWS/files/",
        csv_label="(not used - annotations come from the annotation dir)",
        extra_placeholder="e.g. /data/CHEST-IMAGENOME/",
        extra_dropdown_label="Split",
        extra_dropdown_choices=("all", "train", "valid", "test"),
        extra_dropdown_kwarg="split",
    ),
    "── RSNA Pneumonia ──": _header("CXR"),
    "RSNA Pneumonia": DatasetConfig(
        False,
        "",
        _build_rsna_pneumonia,
        modality="CXR",
        base_dir_placeholder="e.g. ~/Downloads/rsna/",
        csv_placeholder="auto: pneumonia-challenge-annotations-adjudicated-kaggle_2018.json",
    ),
    "RSNA Pneumonia (Kaggle, Train)": DatasetConfig(
        False,
        "BBox CSV (optional)",
        _build_rsna_pneumonia_kaggle_train,
        modality="CXR",
        base_dir_placeholder="e.g. ~/datasets/rsna-pneumonia-detection-challenge/stage_2_train_images/",
        csv_placeholder="auto: stage_2_detailed_class_info.csv",
        extra_placeholder="auto: stage_2_train_labels.csv",
    ),
    "RSNA Pneumonia (Kaggle, Test)": DatasetConfig(
        False,
        "",
        _build_rsna_pneumonia_kaggle_test,
        modality="CXR",
        base_dir_placeholder="e.g. ~/datasets/rsna-pneumonia-detection-challenge/stage_2_test_images/",
    ),
    # ── CT datasets ───────────────────────────────────────────────────────────
    "── CT-RATE ──": _header("CT"),
    "CT-RATE": DatasetConfig(
        True,
        "Metadata CSV (optional)",
        _build_ctrate,
        modality="CT",
        base_dir_placeholder="e.g. /data/CT-RATE/dataset/train_fixed/",
        csv_placeholder="auto: train_predicted_labels.csv",
        extra_placeholder="auto: train_metadata.csv",
    ),
    "── RAD-ChestCT ──": _header("CT"),
    "RAD-ChestCT": DatasetConfig(
        True,
        "Label CSV (optional)",
        _build_radchestct,
        modality="CT",
        extra_field2_label="BBox CSV (optional)",
        base_dir_placeholder="e.g. /data/RAD-ChestCT/images/",
        csv_placeholder="auto: CT_Scan_Metadata_Complete_35747.csv",
        extra_placeholder="auto: imgtrain_Abnormality_and_Location_Labels.csv",
    ),
    "── RSNA Abdominal Trauma ──": _header("CT"),
    "RSNA Abdominal Trauma (Train)": DatasetConfig(
        True,
        "Series metadata CSV (optional)",
        _build_rsna_abdominal_trauma_2023_train,
        modality="CT",
        base_dir_placeholder="e.g. ~/datasets/external/rsna-2023-abdominal-trauma-detection/train_images/",
        csv_placeholder="auto: train_2024.csv",
        extra_placeholder="auto: train_series_meta.csv",
    ),
    "RSNA Abdominal Trauma (Test)": DatasetConfig(
        True,
        "Series metadata CSV (optional)",
        _build_rsna_abdominal_trauma_2023_test,
        modality="CT",
        base_dir_placeholder="e.g. ~/datasets/external/rsna-2023-abdominal-trauma-detection/test_images/",
        csv_placeholder="auto: test_series_meta.csv",
    ),
    "── RSNA PE Detection ──": _header("CT"),
    "RSNA PE Detection (Train)": DatasetConfig(
        True,
        "",
        _build_rsna_pe_detection_train,
        modality="CT",
        base_dir_placeholder="e.g. /data/rsna_pe_dataset/train/",
        csv_placeholder="auto: train.csv",
    ),
    "RSNA PE Detection (Test)": DatasetConfig(
        True,
        "",
        _build_rsna_pe_detection_test,
        modality="CT",
        base_dir_placeholder="e.g. /data/rsna_pe_dataset/test/",
        csv_placeholder="auto: test.csv",
    ),
    "── RSNA 2022 Cervical Spine ──": _header("CT"),
    "RSNA 2022 Cervical Spine (Train)": DatasetConfig(
        True,
        "Segmentations dir (optional, for output_mask)",
        _build_rsna_2022_cervical_spine_train,
        modality="CT",
        base_dir_placeholder="e.g. ~/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_images/",
        csv_placeholder="auto: train.csv",
        extra_placeholder="auto: ../segmentations/",
    ),
    "RSNA 2022 Cervical Spine (Test)": DatasetConfig(
        True,
        "",
        _build_rsna_2022_cervical_spine_test,
        modality="CT",
        base_dir_placeholder="e.g. ~/datasets/external/rsna-2022-cervical-spine-fracture-detection/test_images/",
        csv_placeholder="auto: test.csv",
    ),
    "RSNA 2022 Cervical Spine (BBox)": DatasetConfig(
        True,
        "BBox CSV (optional)",
        _build_rsna_2022_cervical_spine_bbox,
        modality="CT",
        base_dir_placeholder="e.g. ~/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_images/",
        csv_placeholder="auto: train.csv",
        extra_placeholder="auto: train_bounding_boxes.csv",
    ),
    # ── MRI datasets ──────────────────────────────────────────────────────────
    "── RSNA 2024 Lumbar Spine ──": _header("MRI"),
    "RSNA 2024 Lumbar Spine (Train)": DatasetConfig(
        True,
        "Series descriptions CSV (optional)",
        _build_rsna_2024_lumbar_spine_train,
        modality="MRI",
        base_dir_placeholder="e.g. ~/datasets/competitions/rsna-2024-lumbar-spine-degenerative-classification/train_images/",
        csv_placeholder="auto: train.csv",
        extra_placeholder="auto: train_series_descriptions.csv",
        extra_dropdown_label="Series type",
        extra_dropdown_choices=("(all)", "Axial T2", "Sagittal T1", "Sagittal T2/STIR"),
        extra_dropdown_kwarg="series_filter",
    ),
    "RSNA 2024 Lumbar Spine (Test)": DatasetConfig(
        True,
        "Series descriptions CSV (optional)",
        _build_rsna_2024_lumbar_spine_test,
        modality="MRI",
        base_dir_placeholder="e.g. ~/datasets/competitions/rsna-2024-lumbar-spine-degenerative-classification/test_images/",
        csv_placeholder="auto: test_series_descriptions.csv",
        extra_dropdown_label="Series type",
        extra_dropdown_choices=("(all)", "Axial T2", "Sagittal T1", "Sagittal T2/STIR"),
        extra_dropdown_kwarg="series_filter",
    ),
    # ── Radiograph datasets ───────────────────────────────────────────────────
    "── RSNA Bone Age ──": _header("Radiograph"),
    "RSNA Bone Age (Train)": DatasetConfig(
        False,
        "",
        _build_rsna_bone_age_train,
        modality="Radiograph",
        base_dir_placeholder="e.g. ~/datasets/boneage-training-dataset/",
        csv_placeholder="auto: train.csv",
    ),
    "RSNA Bone Age (Val)": DatasetConfig(
        False,
        "",
        _build_rsna_bone_age_val,
        modality="Radiograph",
        base_dir_placeholder="e.g. ~/datasets/Bone Age Validation Set/",
        csv_placeholder="auto: Validation Dataset.csv",
    ),
    # ── VQA datasets ─────────────────────────────────────────────────────────
    "── ROCO ──": _header("VQA"),
    "ROCO (radiology captioning)": DatasetConfig(
        False,
        "",
        _build_roco,
        modality="VQA",
        base_dir_placeholder="e.g. /data/roco-dataset/",
        csv_placeholder="(not required)",
        extra_placeholder="splits filter (default: train,validation,test)",
    ),
    "── GEMeX-VQA ──": _header("VQA"),
    "GEMeX-VQA": DatasetConfig(
        False,
        "GEMeX-VQA data dir (JSONL files)",
        _build_gemex_vqa,
        modality="VQA",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
        csv_placeholder="(not required)",
        extra_placeholder="e.g. ~/datasets/gemex-vqa/",
    ),
}

# Optional dataset plugins. Each package named in the RADHARMONY_APP_PLUGINS
# env var (comma-separated) may expose a callable
# ``register_app_datasets(registry, DatasetConfig, _header, app_dtype)`` that
# appends entries to DATASET_REGISTRY. Lets third-party / private dataset
# packages extend the app without editing it.
for _plugin in os.environ.get("RADHARMONY_APP_PLUGINS", "").split(","):
    _plugin = _plugin.strip()
    if not _plugin:
        continue
    try:
        _mod = importlib.import_module(_plugin)
        _mod.register_app_datasets(DATASET_REGISTRY, DatasetConfig, _header, _APP_DTYPE)
    except Exception as _e:  # noqa: BLE001 — a bad plugin shouldn't kill the app
        print(f"[radharmony] app plugin {_plugin!r} not loaded: {_e}")


def _group_by_modality(registry: dict) -> dict[str, list[str]]:
    """Return ``{modality: [dataset_name, ...]}`` in insertion order."""
    groups: dict[str, list[str]] = {}
    for name, cfg in registry.items():
        groups.setdefault(cfg.modality, []).append(name)
    return groups


def _split_into_families_per_modality(
    registry: dict,
) -> dict[str, dict[str, list[str]]]:
    """Return ``{modality: {family: [variant_registry_key, ...]}}``.

    Each ``── ... ──`` header starts a new family in its modality; subsequent
    non-header keys belong to that family.  Singletons must have a header in
    front of them — a non-header without a preceding header for its modality
    is a registry mistake and raises.
    """
    out: dict[str, dict[str, list[str]]] = {}
    current_family: dict[str, str] = {}
    for name, cfg in registry.items():
        per_mod = out.setdefault(cfg.modality, {})
        if cfg.is_group_header:
            family = name.strip(" ─")
            current_family[cfg.modality] = family
            per_mod.setdefault(family, [])
        else:
            family = current_family.get(cfg.modality)
            if family is None:
                raise ValueError(
                    f"Dataset {name!r} (modality={cfg.modality!r}) has no "
                    "preceding ── family ── header in DATASET_REGISTRY."
                )
            per_mod.setdefault(family, []).append(name)
    return out


_MODALITY_GROUPS = _group_by_modality(DATASET_REGISTRY)
_MODALITY_FAMILIES = _split_into_families_per_modality(DATASET_REGISTRY)


# ---------------------------------------------------------------------------
# Image rendering helpers
# ---------------------------------------------------------------------------


def _img_to_float(img_tensor: torch.Tensor) -> np.ndarray:
    """(1,H,W) tensor in [-1,1] → (H,W) float32 in [0,1]."""
    arr = img_tensor.float().squeeze(0).numpy()
    return np.clip((arr + 1.0) / 2.0, 0.0, 1.0)


def _to_rgb(arr01: np.ndarray) -> np.ndarray:
    """(H,W) float [0,1] → (H,W,3) uint8."""
    g = (arr01 * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _overlay_mask(
    rgb: np.ndarray, mask_tensor: torch.Tensor, alpha: float = 0.45
) -> np.ndarray:
    """Red-tint mask region on an RGB image."""
    mask = mask_tensor.float().squeeze(0).numpy()
    if mask.ndim == 3:  # 3-D volume: take mid-slice
        mask = mask[mask.shape[0] // 2]
    region = mask > 0.5
    if not region.any():
        return rgb
    out = rgb.copy()
    out[region, 0] = np.clip(out[region, 0] * (1 - alpha) + 255 * alpha, 0, 255).astype(
        np.uint8
    )
    out[region, 1] = (out[region, 1] * (1 - alpha)).astype(np.uint8)
    out[region, 2] = (out[region, 2] * (1 - alpha)).astype(np.uint8)
    return out


def _overlay_bbox(
    rgb: np.ndarray,
    bbox,
    spatial: tuple,
    projection_axis: int = 0,
    labels=None,
    as_dots: bool = False,
) -> np.ndarray:
    """Draw green bounding-box rectangle(s) on an RGB image with optional labels.

    Supports both a single normalized bbox (flat list) and multiple
    normalized bboxes (list of lists / 2-D tensor).  All coordinates
    must be in ``[0, 1]`` with format
    ``[dim0_min, dim0_max, dim1_min, dim1_max, ...]``.

    Args:
        rgb: (H, W, 3) uint8 image.
        bbox: A single bbox (flat list/tensor) or multiple bboxes (list of
            lists / 2-D tensor).
        spatial: Spatial shape ``(H, W)`` or ``(D, H, W)`` of the volume.
        projection_axis: Axis collapsed when projecting 3-D → 2-D.
        labels: Optional list of class-name strings parallel to *bbox*.
        as_dots: When True, draw a small filled dot at each bbox center
            instead of a rectangle outline.  Useful for densely overlapping
            bbox sets (e.g. spine fracture / lumbar coord annotations).
    """
    import json as _json

    try:
        import torch

        if isinstance(bbox, torch.Tensor):
            if bbox.numel() == 0:
                return rgb
            bbox = bbox.tolist()
    except ImportError:
        pass

    # Harmonizers that store bbox as a JSON column produce strings here.
    if isinstance(bbox, str):
        try:
            bbox = _json.loads(bbox)
        except (ValueError, TypeError):
            return rgb

    if not bbox:
        return rgb

    # Determine if this is a single bbox (flat) or multiple (list of lists)
    try:
        first = bbox[0]
    except (IndexError, TypeError):
        return rgb
    if isinstance(first, (list, tuple)):
        boxes = [[float(v) for v in b] for b in bbox]
    else:
        boxes = [[float(v) for v in bbox]]

    H, W = rgb.shape[:2]
    # Scale font size to image resolution (roughly 1/30 of image height)
    font_size = max(8, H // 48)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size
        )
    except (OSError, IOError):
        font = ImageFont.load_default()
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    green = (0, 220, 0)
    # Dot radius scales with image height; small enough to keep dense
    # clusters readable, large enough to be visible on big panels.
    dot_radius = max(1, min(3, H // 150))
    for idx, b in enumerate(boxes):
        if any(not (0.0 <= v <= 1.0) for v in b):
            continue
        # _MaskToBbox writes all-zeros as the "no bbox" sentinel.  Without this
        # skip, the inflate-to-min-size logic below renders a tiny stub box at
        # the origin even on samples with no findings.
        if all(v == 0.0 for v in b):
            continue
        if len(spatial) == 2 and len(b) >= 4:
            y_min, y_max = b[0] * H, b[1] * H
            x_min, x_max = b[2] * W, b[3] * W
        elif len(spatial) == 3 and len(b) >= 6:
            remaining = [i for i in range(3) if i != projection_axis]
            row_ax, col_ax = remaining[0], remaining[1]
            y_min = b[row_ax * 2] * H
            y_max = b[row_ax * 2 + 1] * H
            x_min = b[col_ax * 2] * W
            x_max = b[col_ax * 2 + 1] * W
        else:
            continue
        if as_dots:
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)
            if not (0 <= cx < W and 0 <= cy < H):
                continue
            draw.ellipse(
                [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
                fill=green,
                outline=green,
            )
            anchor_x = cx + dot_radius + 3
            anchor_y = cy
        else:
            # Inflate sub-pixel rectangles so thin axes (e.g. lumbar D ≈ 25
            # slices, bbox z-extent ≈ 1 voxel) still render visibly on
            # coronal/sagittal projections.
            min_side = max(4, H // 50)
            if x_max - x_min < min_side:
                cx_b = (x_min + x_max) / 2
                x_min, x_max = cx_b - min_side / 2, cx_b + min_side / 2
            if y_max - y_min < min_side:
                cy_b = (y_min + y_max) / 2
                y_min, y_max = cy_b - min_side / 2, cy_b + min_side / 2
            x0 = max(0, int(round(x_min)))
            y0 = max(0, int(round(y_min)))
            x1 = min(W - 1, int(round(x_max)))
            y1 = min(H - 1, int(round(y_max)))
            if x1 <= x0 or y1 <= y0:
                continue
            draw.rectangle([x0, y0, x1, y1], outline=green, width=2)
            anchor_x = x0
            anchor_y = y0
        # Class label, with a dark backdrop for readability.
        if labels and idx < len(labels):
            label_text = str(labels[idx])
            text_bbox = draw.textbbox((anchor_x, anchor_y), label_text, font=font)
            tw = text_bbox[2] - text_bbox[0]
            th = text_bbox[3] - text_bbox[1]
            if as_dots:
                text_x = min(W - tw - 4, anchor_x)
                text_y = max(0, anchor_y - th // 2)
            else:
                text_x = anchor_x
                text_y = max(0, anchor_y - th - 4)
            draw.rectangle(
                [text_x, text_y, text_x + tw + 4, text_y + th + 2],
                fill=(0, 0, 0, 180),
            )
            draw.text((text_x + 2, text_y), label_text, fill=green, font=font)
    return np.array(pil_img)


# Datasets whose bbox annotations cluster densely in 3-D (every annotated
# slice/level produces a tight cube).  We pick slice indices from the first
# bbox's center and render only that one bbox so the panels stay readable.
_BBOX_FOCUS_FIRST_DATASETS: frozenset = frozenset(
    {
        "RSNA 2022 Cervical Spine (BBox)",
        "RSNA 2024 Lumbar Spine (Train)",
        "RSNA 2024 Lumbar Spine (Test)",
        "Chest ImaGenome (Gold)",
        "Chest ImaGenome (Silver)",
    }
)


def _parse_str_list(v):
    """Parse a JSON-string per-box list into a Python list (pass lists through)."""
    if isinstance(v, str):
        import json as _j
        try:
            return _j.loads(v)
        except (ValueError, TypeError):
            return None
    return v


def _to_box_list(bbox):
    """Normalize a bbox value (tensor / JSON string / list) to a list of boxes."""
    try:
        import torch
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.tolist()
    except ImportError:
        pass
    bbox = _parse_str_list(bbox)
    if not bbox:
        return []
    try:
        first = bbox[0]
    except (IndexError, TypeError):
        return []
    if isinstance(first, (list, tuple)):
        return [list(b) for b in bbox]
    return [list(bbox)]


def _combine_bbox_display_labels(labels, findings):
    """Combine per-box anatomy + findings into overlay strings.

    ``"right lung"`` with findings ``"pneumonia|effusion"`` becomes
    ``"right lung: pneumonia|effusion"``; a finding-free box stays just the
    anatomy.  Returns ``labels`` unchanged when there are no findings (e.g.
    every other bbox dataset), so this is a no-op outside Chest ImaGenome.
    """
    labels = _parse_str_list(labels)
    findings = _parse_str_list(findings)
    if not labels or not findings:
        return labels
    out = []
    for i, a in enumerate(labels):
        f = findings[i] if i < len(findings) else ""
        out.append(f"{a}: {f}" if f else str(a))
    return out


def _sample_to_rgb(
    sample: dict,
    show_mask: bool,
    show_bbox: bool,
    is_3d: bool,
    dataset_name: str = "",
    bbox_idx: int = 0,
) -> np.ndarray:
    """Convert a processed sample dict to a displayable (H, W, 3) uint8 array.

    For 3-D volumes: horizontal strip of axial | coronal | sagittal mid-slices.

    For ``_BBOX_FOCUS_FIRST_DATASETS`` the panels recenter on the bbox at
    position ``bbox_idx`` (modulo n_boxes) — used by the "Next bbox →"
    button to cycle through annotations without re-running the pipeline.
    """
    bbox_focus_first = dataset_name in _BBOX_FOCUS_FIRST_DATASETS
    img = sample["img"]
    raw = img.float().numpy()
    # Reduce to spatial dims: 2-D (H, W) or 3-D (D, H, W).
    # Squeeze leading singletons first, then take channel 0 if still needed
    # (MONAI MetaTensor lazy ops can leak bbox_mask channels into img).
    target_ndim = 3 if is_3d else 2
    while raw.ndim > target_ndim and raw.shape[0] == 1:
        raw = raw.squeeze(0)
    if raw.ndim > target_ndim:
        raw = raw[0]

    if is_3d:
        vol = raw  # (D, H, W)
        d, h, w = vol.shape
        spatial3d = (d, h, w)

        # Mid-slice mask overlay per projection axis.  Reduce the mask the
        # same way we reduce the image so both end up as (D, H, W).  Skip
        # silently when the shape doesn't line up after transforms.
        mask_vol = None
        if show_mask and "mask" in sample:
            m = sample["mask"].float().numpy()
            while m.ndim > 3 and m.shape[0] == 1:
                m = m.squeeze(0)
            if m.ndim > 3:
                m = m[0]
            if m.shape == vol.shape:
                mask_vol = m

        # After OrientationD(IPL)+TransposeD: dim0=I, dim1=P, dim2=L.
        # Axial = slice along I (axis 0), coronal = along P (axis 1), sagittal = along L (axis 2).
        # For focus-first datasets (cervical/lumbar), pick slice indices from
        # the FIRST bbox's center so the displayed slices actually contain it.
        # Other bboxes are dropped from the overlay — keeps the view focused on
        # one annotation rather than projecting every bbox onto every panel.
        slice_idx = [d // 2, h // 2, w // 2]
        bbox_for_overlay = sample.get("bbox") if show_bbox else None
        labels_for_overlay = _combine_bbox_display_labels(
            sample.get("bbox_labels"), sample.get("bbox_findings")
        )
        if bbox_focus_first and show_bbox and "bbox" in sample:
            try:
                bb = sample["bbox"]
                if hasattr(bb, "tolist"):
                    bb_list = bb.tolist()
                else:
                    bb_list = list(bb)
                if (
                    bb_list
                    and isinstance(bb_list[0], (list, tuple))
                    and len(bb_list[0]) >= 6
                ):
                    pick = int(bbox_idx) % len(bb_list)
                    chosen = [float(v) for v in bb_list[pick]]
                    cz = (chosen[0] + chosen[1]) / 2
                    cy = (chosen[2] + chosen[3]) / 2
                    cx = (chosen[4] + chosen[5]) / 2
                    slice_idx = [
                        int(np.clip(round(cz * d), 0, d - 1)),
                        int(np.clip(round(cy * h), 0, h - 1)),
                        int(np.clip(round(cx * w), 0, w - 1)),
                    ]
                    bbox_for_overlay = [chosen]
                    if labels_for_overlay is not None and pick < len(
                        labels_for_overlay
                    ):
                        labels_for_overlay = [labels_for_overlay[pick]]
                    elif labels_for_overlay is not None:
                        labels_for_overlay = None
            except (TypeError, ValueError, IndexError):
                pass

        axes_slices = [
            (np.take(vol, slice_idx[0], axis=0), 0),  # axial    (cut along S)
            (np.take(vol, slice_idx[1], axis=1), 1),  # coronal  (cut along P)
            (np.take(vol, slice_idx[2], axis=2), 2),  # sagittal (cut along L)
        ]
        panels = []
        alpha = 0.45  # match _overlay_mask's tint strength
        for slc, proj_ax in axes_slices:
            arr01 = np.clip((slc + 1.0) / 2.0, 0.0, 1.0)
            p = _to_rgb(arr01)
            if mask_vol is not None:
                ms = np.take(mask_vol, slice_idx[proj_ax], axis=proj_ax)
                region = ms > 0.5
                if region.any():
                    p = p.copy()
                    p[region, 0] = np.clip(
                        p[region, 0] * (1 - alpha) + 255 * alpha, 0, 255
                    ).astype(np.uint8)
                    p[region, 1] = (p[region, 1] * (1 - alpha)).astype(np.uint8)
                    p[region, 2] = (p[region, 2] * (1 - alpha)).astype(np.uint8)
            if show_bbox and bbox_for_overlay is not None:
                p = _overlay_bbox(
                    p,
                    bbox_for_overlay,
                    spatial3d,
                    proj_ax,
                    labels=labels_for_overlay,
                )
            panels.append(p)
        ph = max(p.shape[0] for p in panels)
        pw = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            canvas = np.zeros((ph, pw, 3), dtype=np.uint8)
            canvas[: p.shape[0], : p.shape[1]] = p
            padded.append(canvas)
        sep = np.full((ph, 3, 3), 80, dtype=np.uint8)
        rgb = np.concatenate([padded[0], sep, padded[1], sep, padded[2]], axis=1)
    else:
        arr01 = np.clip((raw + 1.0) / 2.0, 0.0, 1.0)
        rgb = _to_rgb(arr01)
        if show_mask and "mask" in sample:
            rgb = _overlay_mask(rgb, sample["mask"])
        if show_bbox and "bbox" in sample:
            # Anatomy + per-box findings (Chest ImaGenome) as overlay text.
            _lbl = _combine_bbox_display_labels(
                sample.get("bbox_labels"), sample.get("bbox_findings")
            )
            _bbox_for_overlay = sample["bbox"]
            # Focus-first (dense multi-box CXR): show one box + its label at a
            # time, cycled by the "Next bbox →" button.
            if bbox_focus_first:
                _bb_list = _to_box_list(sample["bbox"])
                if _bb_list and isinstance(_bb_list[0], (list, tuple)):
                    _pick = int(bbox_idx) % len(_bb_list)
                    _bbox_for_overlay = [_bb_list[_pick]]
                    if _lbl and _pick < len(_lbl):
                        _lbl = [_lbl[_pick]]
            rgb = _overlay_bbox(
                rgb,
                _bbox_for_overlay,
                raw.shape[-2:],
                labels=_lbl,
            )

    return rgb


# ---------------------------------------------------------------------------
# Transform builders
# ---------------------------------------------------------------------------


def _aug_extremes(aug_name: str, p: dict) -> tuple[dict, dict]:
    """Return (p_min, p_max) — param dicts collapsed to each extreme for
    deterministic side-by-side comparison."""
    lo, hi = dict(p), dict(p)
    if aug_name == "transpose":
        return lo, hi  # deterministic — no meaningful min/max
    if aug_name == "jitter":
        lo["jitter_shift"] = lo["jitter_scale"] = 0.01
    elif aug_name == "noise":
        lo["noise_std"] = 0.01
    elif aug_name == "smooth":
        lo["smooth_min"] = lo["smooth_max"] = p["smooth_min"]
        hi["smooth_min"] = hi["smooth_max"] = p["smooth_max"]
    elif aug_name == "elastic":
        lo["elastic_mag_min"] = lo["elastic_mag_max"] = p["elastic_mag_min"]
        lo["elastic_sigma_min"] = p["elastic_sigma_min"]
        hi["elastic_mag_min"] = hi["elastic_mag_max"] = p["elastic_mag_max"]
        hi["elastic_sigma_min"] = p["elastic_sigma_min"]
    elif aug_name == "affine":
        lo["affine_translate"] = lo["affine_rotate"] = 0.0
        lo["affine_scale"] = lo["affine_shear"] = 0.0
    return lo, hi


def _apply_aug(builder, aug_name: str, p: dict) -> None:
    """Apply a named augmentation to *builder* using parameter dict *p*."""
    if aug_name == "transpose":
        try:
            indices = [int(x.strip()) for x in p["transpose_indices"].split(",")]
        except (ValueError, KeyError):
            indices = [0, 2, 1]
        # Tensors are (C, H, W) in 2-D or (C, D, H, W) in 3-D — validate now
        # so we fail fast with a helpful message instead of deep in MONAI.
        expected = {3, 4}
        if len(indices) not in expected:
            raise ValueError(
                f"Transpose indices {indices} have {len(indices)} entries; "
                "need 3 for 2-D tensors (e.g. 0,2,1) or 4 for 3-D (e.g. 0,1,3,2)."
            )
        builder.with_transpose(indices=indices)
    elif aug_name == "flip":
        builder.with_flip(spatial_axis=p["flip_axis"], prob=1.0)
    elif aug_name == "jitter":
        builder.with_intensity_jitter(
            shift=p["jitter_shift"], scale=p["jitter_scale"], prob=1.0
        )
    elif aug_name == "noise":
        builder.with_gaussian_noise(std=p["noise_std"], prob=1.0)
    elif aug_name == "smooth":
        if isinstance(builder, RadiologyTransform3D):
            # 3-D Gaussian smoothing convolves a kernel of size ~4*sigma+1 per
            # axis, so the work and memory scale with sigma^3.  sigma > 10 on
            # a 112^3 volume can easily exceed the OOM budget (process gets
            # SIGKILL-ed by the OS with no Python traceback).  Fail fast.
            smax = float(p["smooth_max"])
            if smax > 10.0:
                raise ValueError(
                    f"Gaussian Smooth for 3-D volumes: sigma_max={smax:g} "
                    "is unsafe.  Safe range on 112^3 volumes: 0.1–10 "
                    "(default 1.5).  Larger values push the separable "
                    "Gaussian convolution into multi-GB intermediate "
                    "buffers and typically get the process killed by the OS."
                )
        builder.with_gaussian_smooth(
            sigma_range=(p["smooth_min"], p["smooth_max"]), prob=1.0
        )
    elif aug_name == "elastic":
        if isinstance(builder, RadiologyTransform3D):
            # 3-D Rand3DElastic allocates a full-volume displacement field
            # smoothed by a Gaussian of `sigma`.  On a 112^3 volume, sigma > 50
            # or magnitude > 50 can easily push peak RSS past 20 GB and get
            # the process OOM-killed by the OS with no Python traceback.  Fail
            # fast with a clear message instead.
            sigma = float(p["elastic_sigma_min"])
            mag_max = float(p["elastic_mag_max"])
            if sigma > 50 or mag_max > 50:
                raise ValueError(
                    f"Elastic Deformation for 3-D volumes: sigma={sigma:g} "
                    f"and/or magnitude_max={mag_max:g} are unsafe.  Safe ranges "
                    "on 112^3 volumes: sigma 5–30 (default 20), "
                    "magnitude_max 5–30 (default 15).  Larger values allocate "
                    "multi-GB intermediate buffers and typically get the "
                    "process killed by the OS."
                )
            builder.with_elastic_deformation(
                sigma_range=(sigma, sigma),
                magnitude_range=(p["elastic_mag_min"], p["elastic_mag_max"]),
                prob=1.0,
            )
        else:
            builder.with_elastic_deformation(
                spacing=int(p["elastic_sigma_min"]),
                magnitude_range=(p["elastic_mag_min"], p["elastic_mag_max"]),
                prob=1.0,
            )
    elif aug_name == "affine":
        builder.with_affine(
            translate_range=(p["affine_translate"], p["affine_translate"]),
            rotate_range=(p["affine_rotate"],),
            scale_range=(p["affine_scale"], p["affine_scale"]),
            shear_range=(p["affine_shear"],),
            prob=1.0,
        )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def _probe_native_spacing(image_path: str) -> tuple[float, float, float] | None:
    """Native voxel spacing in mm for ``image_path`` (DICOM series dir or
    single file).  Returns ``(x, y, z)`` or ``None`` if unknown.  Reads one
    DICOM file via pydicom for series dirs (fast); falls back to MONAI's
    ITKReader for single-file formats (NIfTI, etc.).  Used to populate the
    "Native: ..." readout under the spacing sliders."""
    if not image_path:
        return None
    # ``image_path`` may contain a leading ``~`` if get_data_dict was called
    # with the user's unexpanded base_dir.  os.path.exists doesn't expand
    # tilde, so probe failures used to look like "spacing not available" on
    # otherwise-fine datasets — expand defensively.
    image_path = os.path.expanduser(image_path)
    if not os.path.exists(image_path):
        return None

    if os.path.isdir(image_path):
        try:
            import pydicom

            for fname in sorted(os.listdir(image_path)):
                if fname.lower().endswith((".dcm", ".dicom")):
                    ds = pydicom.dcmread(
                        os.path.join(image_path, fname), stop_before_pixels=True
                    )
                    px, py = ds.PixelSpacing
                    pz = getattr(ds, "SliceThickness", None) or getattr(
                        ds, "SpacingBetweenSlices", None
                    )
                    if pz is None:
                        return None
                    return (float(px), float(py), float(pz))
        except Exception:
            return None
        return None

    try:
        import monai.transforms as _mt

        loader = _mt.LoadImage(reader="ITKReader", image_only=False)
        _img, meta = loader(image_path)
        if "pixdim" in meta:
            pd = meta["pixdim"]
            return (float(pd[1]), float(pd[2]), float(pd[3]))
        if "affine" in meta:
            aff = np.asarray(meta["affine"])
            sp = np.linalg.norm(aff[:3, :3], axis=0)
            return (float(sp[0]), float(sp[1]), float(sp[2]))
    except Exception:
        return None
    return None


def _format_native_spacing(spacing: tuple[float, float, float] | None) -> str:
    """User-facing label for the "Native: ..." caption above the sliders."""
    if spacing is None:
        return "Native spacing: _not available for this dataset_"
    x, y, z = spacing
    return f"Native spacing: **{x:.2f} × {y:.2f} × {z:.2f} mm**"


def _make_state(
    data_dicts,
    label_cols,
    reg_cols,
    available_keys,
    is_3d,
    dataset_name="",
    native_spacing=None,
):
    return {
        "data_dicts": data_dicts,
        "label_cols": label_cols,
        "reg_cols": reg_cols,
        "available_keys": available_keys,
        "is_3d": is_3d,
        "dataset_name": dataset_name,
        "native_spacing": native_spacing,
    }


def load_dataset(
    dataset_name,
    base_dir,
    csv_path,
    extra_field,
    extra_field2,
    extra_dropdown,
    cache_dir,
    want_cls,
    want_mask,
    want_report,
    want_bbox,
    want_reg,
    progress=gr.Progress(track_tqdm=True),
):
    """Load any registered dataset. Dispatches via DATASET_REGISTRY."""
    # All four early-return paths must return 3 values to match the binding
    # in 3-D tabs: outputs=[state, load_status, native_spacing_md].  The
    # markdown placeholder is computed once here and reused.
    _no_native = _format_native_spacing(None)

    base_dir = (base_dir or "").strip()
    if not base_dir:
        return None, "Please provide a base image directory.", _no_native

    # Pre-flight: catch typos like a missing leading "~/" or "/mnt/"
    # prefix before they fall through into the per-build-function "<file> is
    # required" messages, which can't distinguish "directory missing" from
    # "expected file missing inside an otherwise-fine directory".
    if not os.path.isdir(os.path.expanduser(base_dir)):
        return None, f"Base directory not found: {base_dir}", _no_native

    config = DATASET_REGISTRY.get(dataset_name)
    if config is None:
        return None, f"Unknown dataset: {dataset_name!r}", _no_native
    if config.is_group_header:
        return None, "Select a dataset from the list (not a group header).", _no_native

    # Forward the dropdown value as a dataset-specific kwarg if the config
    # declares one.  "(all)"/empty means "no filter".
    build_flags = dict(
        output_cls=want_cls,
        output_mask=want_mask,
        output_report=want_report,
        output_bbox=want_bbox,
        output_reg=want_reg,
    )
    if config.extra_dropdown_kwarg:
        val = (extra_dropdown or "").strip()
        build_flags[config.extra_dropdown_kwarg] = None if val in ("", "(all)") else val

    try:
        progress(0, desc="Building dataset…")
        ds_obj, err = config.build(
            base_dir,
            (csv_path or "").strip(),
            (extra_field or "").strip(),
            (extra_field2 or "").strip(),
            (cache_dir or "").strip() or None,
            **build_flags,
        )
        if err:
            return None, err, _no_native

        progress(0.3, desc="Verifying image files…")
        missing = ds_obj.verify_images()
        n_missing = len(missing)

        progress(0.5, desc="Harmonizing metadata…")
        df = ds_obj.get_harmonized_df()
        progress(0.8, desc="Preparing data dicts…")
        data_dicts = get_data_dict(
            df, base_dir, "image_path", ds_obj._cols, num_cores=1
        )

        available_keys = {"img"}
        if data_dicts:
            for k in ["cls", "mask", "report", "qa_text", "bbox", "bbox_labels", "bbox_findings", "reg"]:
                if k in data_dicts[0]:
                    available_keys.add(k)

        # Probe native voxel spacing of the first sample (3-D only — meaningless
        # for 2-D radiographs).  Powers the "Native: ..." caption and the
        # "Match native" / "Half native" preset buttons under the spacing
        # sliders.  None on failure / 2-D / unsupported format.
        native_spacing = None
        if config.is_3d and data_dicts:
            # data_dicts[0]['img'] inherits whatever base_dir form the user
            # typed — including a leading "~".  os.path.isabs returns False
            # for tilde-paths, so naively joining with base_dir produced a
            # nonsense "~/.../train/~/.../train/..." string.  Expand both
            # sides up front so the rest of the logic works on real absolute
            # paths.
            first_path = os.path.expanduser(data_dicts[0].get("img", ""))
            if first_path:
                if os.path.isabs(first_path):
                    full = first_path
                else:
                    full = os.path.join(os.path.expanduser(base_dir or ""), first_path)
                native_spacing = _probe_native_spacing(full)

        state = _make_state(
            data_dicts,
            ds_obj.LABEL_COLS,
            getattr(ds_obj, "REG_COLS", []),
            available_keys,
            is_3d=config.is_3d,
            dataset_name=dataset_name,
            native_spacing=native_spacing,
        )
        progress(1.0)
        msg = f"Loaded {len(data_dicts)} samples.  Keys: {sorted(available_keys)}"
        if n_missing:
            msg += f"  ({n_missing} rows dropped — missing image files)"
        return state, msg, _format_native_spacing(native_spacing)
    except Exception as exc:
        return None, f"Error: {exc}", _format_native_spacing(None)


def _update_extra_fields(dataset_name: str):
    """Return gr.updates for base_dir, csv_path, both extra textboxes, and the optional dropdown."""
    config = DATASET_REGISTRY.get(dataset_name)
    if config is None or config.is_group_header:
        config = None
    label1 = config.extra_field_label if config else "Extra field"
    label2 = config.extra_field2_label if config else ""
    ph_base = config.base_dir_placeholder if config else ""
    ph_csv = config.csv_placeholder if config else ""
    csv_label = config.csv_label if config else "CSV path (optional)"
    ph1 = config.extra_placeholder if config else ""
    ph2 = config.extra_placeholder2 if config else ""
    dd_label = config.extra_dropdown_label if config else ""
    dd_choices = list(config.extra_dropdown_choices) if config else []
    dd_value = dd_choices[0] if dd_choices else None
    return (
        gr.update(placeholder=ph_base, value=""),
        gr.update(label=csv_label, placeholder=ph_csv, value=""),
        gr.update(label=label1, placeholder=ph1, visible=bool(label1)),
        gr.update(label=label2, placeholder=ph2, visible=bool(label2)),
        gr.update(
            label=dd_label,
            choices=dd_choices,
            value=dd_value,
            visible=bool(dd_label),
        ),
    )


# ---------------------------------------------------------------------------
# Report helper
# ---------------------------------------------------------------------------


def _read_report(value) -> str:
    """Return report text: read file if value is a path, else return as-is."""
    s = str(value)
    if os.path.isfile(s):
        try:
            with open(s) as f:
                return f.read().strip()
        except Exception:
            return s
    return s


# ---------------------------------------------------------------------------
# Transform code generation
# ---------------------------------------------------------------------------


def _aug_params_str(aug_name: str, p: dict) -> str:
    """Compact human-readable summary of *aug_name* params."""
    if aug_name == "flip":
        return f"axis={p['flip_axis']}"
    if aug_name == "jitter":
        return f"shift={p['jitter_shift']:.2f}  scale={p['jitter_scale']:.2f}"
    if aug_name == "noise":
        return f"std={p['noise_std']:.3f}"
    if aug_name == "smooth":
        lo, hi = p["smooth_min"], p["smooth_max"]
        return f"σ={lo:.1f}" if lo == hi else f"σ=[{lo:.1f}, {hi:.1f}]"
    if aug_name == "elastic":
        sp = int(p["elastic_sigma_min"])
        mlo, mhi = p["elastic_mag_min"], p["elastic_mag_max"]
        mag = f"{mlo:.0f}" if mlo == mhi else f"[{mlo:.0f}, {mhi:.0f}]"
        return f"spacing={sp}  mag={mag}"
    if aug_name == "affine":
        return (
            f"t={p['affine_translate']:.0f}px  "
            f"r={p['affine_rotate']:.2f}rad  "
            f"s={p['affine_scale']:.2f}  "
            f"sh={p['affine_shear']:.2f}"
        )
    if aug_name == "transpose":
        return f"indices=[{p['transpose_indices']}]"
    return ""


def _build_transform_code(
    is_3d: bool,
    img_size: int,
    pad: bool,
    enabled_augs: list,
    aug_params: dict,
    avail_keys: set,
    hu_window: bool = False,
    hu_min: float = -1000.0,
    hu_max: float = 1000.0,
    dataset_name: str = "",
    pixdim: tuple[float, float, float] | None = None,
) -> str:
    """Generate a copy-pasteable Python snippet for the current transform config."""
    cls = "RadiologyTransform3D" if is_3d else "RadiologyTransform2D"
    p = aug_params
    lines = [f"from radharmony.dataset.transforms import {cls}", ""]

    keys_repr = "{" + ", ".join(f'"{k}"' for k in sorted(avail_keys)) + "}"
    lines.append("transform = (")
    if is_3d:
        hw_str = f"({hu_min}, {hu_max})" if hu_window else "None"
        base_transpose = dataset_name != "RAD-ChestCT"
        lines += [
            f"    {cls}(",
            f"        img_size={img_size},",
            f"        output_keys={keys_repr},",
            f"        pad={pad},",
            f"        hu_window={hw_str},",
        ]
        if not base_transpose:
            lines.append(f"        base_transpose=False,")
        lines.append(f"    )")
        if pixdim is not None:
            lines.append(f"    .with_spacing(({pixdim[0]}, {pixdim[1]}, {pixdim[2]}))")
    else:
        lines += [
            f"    {cls}(",
            f"        img_size={img_size},",
            f"        output_keys={keys_repr},",
            f"        pad={pad},",
            f"    )",
        ]

    for aug in enabled_augs:
        if aug == "flip":
            lines.append(f"    .with_flip(spatial_axis={p['flip_axis']}, prob=0.5)")
        elif aug == "jitter":
            lines.append(
                f"    .with_intensity_jitter(shift={p['jitter_shift']}, scale={p['jitter_scale']}, prob=0.5)"
            )
        elif aug == "noise":
            lines.append(f"    .with_gaussian_noise(std={p['noise_std']}, prob=0.5)")
        elif aug == "smooth":
            lines.append(
                f"    .with_gaussian_smooth(sigma_range=({p['smooth_min']}, {p['smooth_max']}), prob=0.5)"
            )
        elif aug == "elastic":
            if is_3d:
                lines += [
                    f"    .with_elastic_deformation(",
                    f"        sigma_range=({p['elastic_sigma_min']}, {p['elastic_sigma_min']}),",
                    f"        magnitude_range=({p['elastic_mag_min']}, {p['elastic_mag_max']}),",
                    f"        prob=0.5,",
                    f"    )",
                ]
            else:
                lines += [
                    f"    .with_elastic_deformation(",
                    f"        spacing={int(p['elastic_sigma_min'])},",
                    f"        magnitude_range=({p['elastic_mag_min']}, {p['elastic_mag_max']}),",
                    f"        prob=0.5,",
                    f"    )",
                ]
        elif aug == "affine":
            lines += [
                f"    .with_affine(",
                f"        translate_range=({p['affine_translate']}, {p['affine_translate']}),",
                f"        rotate_range=({p['affine_rotate']},),",
                f"        scale_range=({p['affine_scale']}, {p['affine_scale']}),",
                f"        shear_range=({p['affine_shear']},),",
                f"        prob=0.5,",
                f"    )",
            ]
        elif aug == "transpose":
            try:
                indices = [int(x.strip()) for x in p["transpose_indices"].split(",")]
            except (ValueError, KeyError):
                indices = [0, 2, 1]
            lines.append(f"    .with_transpose(indices={indices})")

    lines.append("    .get_transform()")
    lines.append(")")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Heavy-aug OOM budget (3-D pipeline)
# ---------------------------------------------------------------------------

# MONAI's heavy spatial transforms (Gaussian Smooth, Elastic, Affine) each hold
# multiple volume-sized float32 buffers + a deformation grid in scratch.  Stack
# 2+ on a large volume and the process can OOM with no traceback.  This budget
# lets the limit track the active CUDA device's free VRAM at click time instead
# of a hardcoded threshold.  Tune upward if the lab GPU consistently flags safe
# combos as risky; downward if you see real OOMs sneak past the guard.
_HEAVY_AUG_VOXEL_BUDGET_PER_GB = 2_000_000


def _free_vram_gb() -> float:
    """Free VRAM in GB on the active CUDA device, or 0.0 if no CUDA."""
    if not torch.cuda.is_available():
        return 0.0
    try:
        free, _total = torch.cuda.mem_get_info()
        return free / (1024**3)
    except Exception:
        return 0.0


def _heavy_aug_budget_violation(
    img_size: int, n_heavy: int, free_gb: float
) -> str | None:
    """Return None if the (img_size, n_heavy) combo fits the VRAM budget,
    else a user-facing reason string.  Falls back to the original conservative
    rule (192³ + 2 heavy → block) when no CUDA is detected."""
    if n_heavy == 0:
        return None
    if free_gb <= 0:
        if img_size >= 192 and n_heavy >= 2:
            return (
                f"Volume {img_size}³ with {n_heavy} heavy spatial augs is likely "
                "to OOM on a CPU/non-CUDA pipeline."
            )
        return None
    voxel_aug_load = n_heavy * (img_size**3)
    budget = _HEAVY_AUG_VOXEL_BUDGET_PER_GB * free_gb
    if voxel_aug_load > budget:
        return (
            f"Volume {img_size}³ with {n_heavy} heavy spatial augs needs "
            f"~{voxel_aug_load / 1_000_000:.0f}M voxel-augs but only "
            f"~{budget / 1_000_000:.0f}M fit the {free_gb:.1f} GB free VRAM "
            "budget."
        )
    return None


# ---------------------------------------------------------------------------
# Core runner (shared by 2-D and 3-D tabs)
# ---------------------------------------------------------------------------


def _run(
    ds_state,
    img_size,
    pad,
    show_mask,
    show_bbox,
    flip,
    flip_axis,
    jitter,
    jitter_shift,
    jitter_scale,
    noise,
    noise_std,
    smooth,
    smooth_min,
    smooth_max,
    elastic,
    elastic_sigma_min,
    elastic_mag_min,
    elastic_mag_max,
    affine,
    affine_translate,
    affine_rotate,
    affine_scale,
    affine_shear,
    transpose,
    transpose_indices,
    safety_override,
    hu_window=False,
    hu_min=-1000.0,
    hu_max=1000.0,
    pixdim=None,
    reuse_sample=None,
):
    if ds_state is None:
        return (
            [],
            "No dataset loaded — click Load Dataset first.",
            "",
            "",
            "",
            "",
            "",
            None,
            [],
        )

    data_dicts = ds_state["data_dicts"]
    label_cols = ds_state["label_cols"]
    reg_cols = ds_state.get("reg_cols", []) or []
    avail_keys = ds_state["available_keys"]
    is_3d = ds_state["is_3d"]
    dataset_name = ds_state.get("dataset_name", "")

    sample_dict = (
        reuse_sample if reuse_sample is not None else random.choice(data_dicts)
    )
    qa_text = sample_dict.get("qa_text", "") or ""

    aug_params = dict(
        flip_axis=int(flip_axis),
        jitter_shift=jitter_shift,
        jitter_scale=jitter_scale,
        noise_std=noise_std,
        smooth_min=smooth_min,
        smooth_max=smooth_max,
        elastic_sigma_min=elastic_sigma_min,
        elastic_mag_min=elastic_mag_min,
        elastic_mag_max=elastic_mag_max,
        affine_translate=affine_translate,
        affine_rotate=affine_rotate,
        affine_scale=affine_scale,
        affine_shear=affine_shear,
        transpose_indices=transpose_indices,
    )

    enabled_augs = (
        (["flip"] if flip else [])
        + (["jitter"] if jitter else [])
        + (["noise"] if noise else [])
        + (["smooth"] if smooth else [])
        + (["elastic"] if elastic else [])
        + (["affine"] if affine else [])
        + (["transpose"] if transpose else [])
    )

    # Volume-size × heavy-augs pre-flight (3-D only).  Budget tracks the live
    # free VRAM (see _heavy_aug_budget_violation) so the limit follows the
    # actual hardware instead of a hardcoded threshold.  The override checkbox
    # downgrades the block to a warning toast for power users on hardware the
    # heuristic underestimates.
    _heavy = {"smooth", "elastic", "affine"}
    _heavy_enabled = [a for a in enabled_augs if a in _heavy]
    if is_3d and _heavy_enabled:
        free_gb = _free_vram_gb()
        reason = _heavy_aug_budget_violation(img_size, len(_heavy_enabled), free_gb)
        if reason:
            tail = (
                f" (augs: {', '.join(_heavy_enabled)}). Reduce volume size, "
                "drop one of {Gaussian Smooth, Elastic, Affine}, or enable "
                "**Override OOM safety guard** below to proceed anyway."
            )
            if safety_override:
                gr.Warning(f"Override active — proceeding despite OOM risk. {reason}")
            else:
                raise gr.Error(reason + tail)

    def make_transform(augs_to_apply: list, p_override: dict | None = None):
        if is_3d:
            hw = (hu_min, hu_max) if hu_window else None
            b = RadiologyTransform3D(
                img_size=img_size,
                output_keys=avail_keys,
                pad=pad,
                hu_window=hw,
                base_transpose=dataset_name != "RAD-ChestCT",
            )
            if pixdim is not None:
                b.with_spacing(pixdim)
        else:
            b = RadiologyTransform2D(img_size=img_size, output_keys=avail_keys, pad=pad)
        if "bbox" in avail_keys:
            b.with_bbox_as_mask()
        for name in augs_to_apply:
            _apply_aug(b, name, p_override if p_override is not None else aug_params)
        return b.get_transform()

    # Original (no augmentation)
    try:
        base_sample = make_transform([])(sample_dict.copy())
    except Exception as exc:
        return [], f"Error loading sample: {exc}", "", "", "", "", "", None, []

    # Post-transform samples cached alongside the gallery so the "Next bbox →"
    # button can re-render with a different bbox index without re-running the
    # transform pipeline.  Subset to the keys _sample_to_rgb actually reads —
    # only include keys whose value is present (else the `"mask" in sample`
    # check downstream falsely picks up None values).
    def _subset(s: dict) -> dict:
        out = {"img": s["img"]}
        for k in ("mask", "bbox", "bbox_labels", "bbox_findings"):
            if k in s and s[k] is not None:
                out[k] = s[k]
        return out

    gallery: list[tuple[np.ndarray, str]] = []
    gallery_samples: list[tuple[str, dict]] = []
    gallery.append(
        (
            _sample_to_rgb(base_sample, show_mask, show_bbox, is_3d, dataset_name),
            "Original",
        )
    )
    gallery_samples.append(("Original", _subset(base_sample)))

    # Tensor-shape + intensity summary for the "Image size & intensity" box.
    # Spatial dims only (strip the leading channel dim) so "224 x 224" /
    # "112 x 112 x 112" read at a glance.
    _img = base_sample["img"]
    _spatial = tuple(int(s) for s in _img.shape[1:])
    _min = float(_img.min())
    _max = float(_img.max())
    image_stats = (
        f"Size: {' x '.join(str(s) for s in _spatial)}"
        f"   |   Intensity: [{_min:.3f}, {_max:.3f}]"
    )

    # Min and max per augmentation
    for aug_name in enabled_augs:
        p_lo, p_hi = _aug_extremes(aug_name, aug_params)
        label = aug_name.capitalize()
        for p_ext, suffix in ((p_lo, "Min"), (p_hi, "Max")):
            try:
                s = make_transform([aug_name], p_ext)(sample_dict.copy())
                caption = f"{label} ({suffix}) — {_aug_params_str(aug_name, p_ext)}"
                gallery.append(
                    (
                        _sample_to_rgb(s, show_mask, show_bbox, is_3d, dataset_name),
                        caption,
                    )
                )
                gallery_samples.append((caption, _subset(s)))
            except Exception as exc:
                blank = np.zeros((64, 64, 3), dtype=np.uint8)
                gallery.append((blank, f"{label} ({suffix}) ERROR: {exc}"))

    # Compound (all enabled augs together)
    if len(enabled_augs) > 1:
        try:
            s = make_transform(enabled_augs)(sample_dict.copy())
            gallery.append(
                (
                    _sample_to_rgb(s, show_mask, show_bbox, is_3d, dataset_name),
                    "Compound",
                )
            )
            gallery_samples.append(("Compound", _subset(s)))
        except Exception as exc:
            blank = np.zeros((64, 64, 3), dtype=np.uint8)
            gallery.append((blank, f"Compound ERROR: {exc}"))

    # Metadata outputs.  Both classification (cls → positive labels) and
    # regression (reg → "col=value" pairs) share the Labels textbox; when
    # a dataset exposes both (rare, but allowed) the classification summary
    # comes first and regression values follow on a new line.
    label_parts: list[str] = []
    if "cls" in base_sample and label_cols:
        vals = base_sample["cls"].float().tolist()
        pos = []
        for i, v in enumerate(vals):
            if not v > 0.5:
                continue
            grade = int(round(v))
            pos.append(f"{label_cols[i]} ({grade})" if grade > 1 else label_cols[i])
        label_parts.append(", ".join(pos) if pos else "No finding")
    if "reg" in base_sample and reg_cols:
        vals = base_sample["reg"].float().tolist()
        label_parts.append(
            ", ".join(f"{reg_cols[i]}={v:.3f}" for i, v in enumerate(vals))
        )
    labels_text = "\n".join(label_parts)

    # bbox-only datasets (no cls) — surface per-box labels in the Labels panel.
    # Chest ImaGenome: "anatomy: findings" per annotated region; other bbox sets
    # (no findings column) fall back to the list of box class labels.
    if not labels_text and "bbox_labels" in base_sample:
        _blabels = _parse_str_list(base_sample.get("bbox_labels")) or []
        _bfind = _parse_str_list(base_sample.get("bbox_findings"))
        if _bfind is not None:
            _lines, _n_plain = [], 0
            for _i, _a in enumerate(_blabels):
                _f = _bfind[_i] if _i < len(_bfind) else ""
                if _f:
                    _lines.append(f"{_a}: {_f}")
                else:
                    _n_plain += 1
            if _lines:
                labels_text = "\n".join(_lines)
                if _n_plain:
                    labels_text += f"\n(+{_n_plain} regions with no finding)"
            elif _blabels:
                labels_text = f"No findings ({len(_blabels)} regions)"
        elif _blabels:
            labels_text = ", ".join(dict.fromkeys(str(x) for x in _blabels))

    report_text = ""
    if "report" in base_sample:
        report_text = _read_report(base_sample["report"])

    code_snippet = _build_transform_code(
        is_3d=is_3d,
        img_size=img_size,
        pad=pad,
        enabled_augs=enabled_augs,
        aug_params=aug_params,
        avail_keys=avail_keys,
        hu_window=hu_window,
        hu_min=hu_min,
        hu_max=hu_max,
        dataset_name=dataset_name,
        pixdim=pixdim,
    )

    return (
        gallery,
        sample_dict.get("img", ""),
        image_stats,
        labels_text,
        report_text,
        qa_text,
        code_snippet,
        sample_dict,
        gallery_samples,
    )


def run_2d(ds_state, img_size, pad, show_mask, show_bbox, *aug_inputs):
    results = _run(ds_state, img_size, pad, show_mask, show_bbox, *aug_inputs)
    # results: gallery, info, image_stats, labels, report, qa_text, code, sample_dict, gallery_samples
    return *results, gr.update(interactive=results[7] is not None)


def run_2d_same(
    ds_state, last_sample, img_size, pad, show_mask, show_bbox, *aug_inputs
):
    if last_sample is None:
        return (
            [],
            "No sample yet — click 'Run on random sample' first.",
            "",
            "",
            "",
            "",
            "",
            None,
            [],
        )
    results = _run(
        ds_state,
        img_size,
        pad,
        show_mask,
        show_bbox,
        *aug_inputs,
        reuse_sample=last_sample,
    )
    return results  # 8 values, no need to update button


def run_3d(
    ds_state,
    img_size,
    pad,
    show_mask,
    show_bbox,
    hu_window,
    hu_min,
    hu_max,
    iso_spacing,
    pixdim_x,
    pixdim_y,
    pixdim_z,
    *aug_inputs,
):
    pixdim = (
        (float(pixdim_x), float(pixdim_y), float(pixdim_z)) if iso_spacing else None
    )
    results = _run(
        ds_state,
        img_size,
        pad,
        show_mask,
        show_bbox,
        *aug_inputs,
        hu_window=hu_window,
        hu_min=hu_min,
        hu_max=hu_max,
        pixdim=pixdim,
    )
    return *results, gr.update(interactive=results[7] is not None)


def run_3d_same(
    ds_state,
    last_sample,
    img_size,
    pad,
    show_mask,
    show_bbox,
    hu_window,
    hu_min,
    hu_max,
    iso_spacing,
    pixdim_x,
    pixdim_y,
    pixdim_z,
    *aug_inputs,
):
    if last_sample is None:
        return (
            [],
            "No sample yet — click 'Run on random sample' first.",
            "",
            "",
            "",
            "",
            "",
            None,
            [],
        )
    pixdim = (
        (float(pixdim_x), float(pixdim_y), float(pixdim_z)) if iso_spacing else None
    )
    return _run(
        ds_state,
        img_size,
        pad,
        show_mask,
        show_bbox,
        *aug_inputs,
        hu_window=hu_window,
        hu_min=hu_min,
        hu_max=hu_max,
        pixdim=pixdim,
        reuse_sample=last_sample,
    )


# ---------------------------------------------------------------------------
# Next-bbox cycling — cervical / lumbar spine only.
# ---------------------------------------------------------------------------


def _next_btn_should_be_active(ds_state, gallery_samples) -> bool:
    """True when the loaded dataset is in _BBOX_FOCUS_FIRST_DATASETS *and*
    a sample with at least one bbox has already been rendered."""
    if not ds_state or not gallery_samples:
        return False
    if ds_state.get("dataset_name", "") not in _BBOX_FOCUS_FIRST_DATASETS:
        return False
    first_bbox = gallery_samples[0][1].get("bbox")
    if first_bbox is None:
        return False
    try:
        return len(first_bbox) > 0
    except TypeError:
        return False


def _next_btn_interactive(ds_state, gallery_samples):
    active = _next_btn_should_be_active(ds_state, gallery_samples)
    return gr.update(visible=active, interactive=active)


def _next_btn_state_after_run(ds_state, gallery_samples):
    """Reset bbox index to 0 and refresh button visibility/interactivity after a fresh run."""
    active = _next_btn_should_be_active(ds_state, gallery_samples)
    return 0, gr.update(visible=active, interactive=active)


def _on_next_bbox(gallery_samples, bbox_idx, show_mask, show_bbox, ds_state):
    """Re-render the gallery with the next bbox in focus.  Pure numpy work —
    no transform pipeline re-run, no augmentation re-rolling."""
    if not gallery_samples or not ds_state:
        return [], bbox_idx
    dataset_name = ds_state.get("dataset_name", "")
    is_3d = ds_state.get("is_3d", False)
    first_bbox = gallery_samples[0][1].get("bbox")
    try:
        n = len(first_bbox) if first_bbox is not None else 0
    except TypeError:
        n = 0
    if n <= 0:
        return [
            (_sample_to_rgb(s, show_mask, show_bbox, is_3d, dataset_name), c)
            for c, s in gallery_samples
        ], bbox_idx
    new_idx = (int(bbox_idx) + 1) % n
    new_gallery = [
        (
            _sample_to_rgb(
                s, show_mask, show_bbox, is_3d, dataset_name, bbox_idx=new_idx
            ),
            c,
        )
        for c, s in gallery_samples
    ]
    return new_gallery, new_idx


# ---------------------------------------------------------------------------
# Shared UI component factory
# ---------------------------------------------------------------------------


def aug_controls(is_3d: bool = False):
    """Render augmentation checkboxes + sliders; return tuple of components.

    Args:
        is_3d: Adds axis=2 (depth) as a flip option. 2-D tensors only have
            spatial axes 0 and 1, so passing ``True`` only makes sense in the
            3-D tab.
    """
    with gr.Accordion(
        "② Augmentations  *(each runs at prob=1.0 when enabled)*",
        open=True,
        elem_classes=["aug-accordion"],
    ):
        with gr.Group():
            with gr.Row():
                flip = gr.Checkbox(label="Flip", value=False)
                if is_3d:
                    flip_axis = gr.Radio(
                        [0, 1, 2],
                        value=1,
                        label="Axis (0=through-plane, 1=up-down, 2=left-right)",
                    )
                else:
                    flip_axis = gr.Radio(
                        [0, 1], value=0, label="Axis (0=up-down, 1=left-right)"
                    )

        with gr.Group():
            with gr.Row():
                jitter = gr.Checkbox(label="Intensity Jitter", value=False)
                jitter_shift = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Shift")
                jitter_scale = gr.Slider(0.0, 1.0, value=0.1, step=0.01, label="Scale")

        with gr.Group():
            with gr.Row():
                noise = gr.Checkbox(label="Gaussian Noise", value=False)
                noise_std = gr.Slider(0.01, 1.0, value=0.05, step=0.01, label="Std")

        with gr.Group():
            with gr.Row():
                smooth = gr.Checkbox(label="Gaussian Smooth", value=False)
                smooth_min = gr.Slider(
                    0.1, 10.0, value=0.5, step=0.1, label="Sigma min"
                )
                smooth_max = gr.Slider(
                    0.1, 20.0, value=1.5, step=0.1, label="Sigma max"
                )

        with gr.Group():
            with gr.Row():
                elastic = gr.Checkbox(label="Elastic Deformation", value=False)
                elastic_sig_min = gr.Slider(
                    1.0,
                    200.0,
                    value=20.0,
                    step=1.0,
                    label="Spacing / Sigma min (2D: spacing px, 3D: sigma min)",
                )
            with gr.Row():
                elastic_mag_min = gr.Slider(
                    1, 200, value=5, step=1, label="Magnitude min"
                )
                elastic_mag_max = gr.Slider(
                    1, 500, value=15, step=1, label="Magnitude max"
                )

        with gr.Group():
            with gr.Row():
                affine = gr.Checkbox(label="Affine", value=False)
                affine_translate = gr.Slider(
                    0.0, 200.0, value=10.0, step=1.0, label="Translate (px)"
                )
                affine_rotate = gr.Slider(
                    0.0, 3.14, value=0.17, step=0.01, label="Rotate (rad, ~10°=0.17)"
                )
            with gr.Row():
                affine_scale = gr.Slider(
                    0.0, 1.0, value=0.1, step=0.01, label="Scale range"
                )
                affine_shear = gr.Slider(
                    0.0, 1.0, value=0.05, step=0.01, label="Shear range"
                )

        with gr.Group():
            with gr.Row():
                transpose = gr.Checkbox(label="Transpose", value=False)
                # 2-D tensors are (C, H, W) → 3-axis permutation; 3-D tensors
                # are (C, D, H, W) → 4-axis. Sending a 3-axis permutation to
                # a 4-D tensor crashes MONAI's Transposed with "input.dim() = 4
                # is not equal to len(dims) = 3".
                _default_indices = "0,1,3,2" if is_3d else "0,2,1"
                transpose_indices = gr.Textbox(
                    value=_default_indices,
                    label="Axis indices (comma-separated, e.g. 0,2,1 for 2D or 0,1,3,2 for 3D)",
                )

        # Power-user escape hatch for the heavy-aug VRAM budget.  Off by
        # default; on means the budget violation downgrades to a warning toast
        # instead of blocking the run.  Surfaced only on the 3-D tab since the
        # guard itself is 3-D-only.
        if is_3d:
            with gr.Group():
                with gr.Row():
                    # No info= helper or trailing caption — matches the other
                    # checkboxes in this accordion (label-only).  The "(advanced)"
                    # suffix flags it as a power-user toggle; the OOM error toast
                    # explains when to enable it at the moment it fires.
                    safety_override = gr.Checkbox(
                        label="Override OOM safety guard (advanced)",
                        value=False,
                    )
        else:
            safety_override = gr.Checkbox(visible=False, value=False)

    return (
        flip,
        flip_axis,
        jitter,
        jitter_shift,
        jitter_scale,
        noise,
        noise_std,
        smooth,
        smooth_min,
        smooth_max,
        elastic,
        elastic_sig_min,
        elastic_mag_min,
        elastic_mag_max,
        affine,
        affine_translate,
        affine_rotate,
        affine_scale,
        affine_shear,
        transpose,
        transpose_indices,
        safety_override,
    )


def _dataset_panel(families: dict[str, list[str]]):
    """Render the dataset input panel for one tab with cascaded family/variant dropdowns.

    Returns a tuple of components used by the Load button:
    (family_dd, variant_dd, base_dir, csv_path, extra_field, extra_field2,
     extra_dropdown, cache_dir, want_cls, want_mask, want_report, want_bbox,
     want_reg, load_btn, load_status)
    """
    family_names = list(families.keys())
    first_family = family_names[0]
    first_variants = families[first_family]
    first_variant = first_variants[0]
    _first = DATASET_REGISTRY[first_variant]

    with gr.Group():
        gr.Markdown("## ① Load Dataset")
        family_dd = gr.Dropdown(
            family_names, value=first_family, label="Dataset family"
        )
        variant_dd = gr.Dropdown(
            first_variants,
            value=first_variant,
            label="Variant",
            visible=len(first_variants) > 1,
        )
        base_dir = gr.Textbox(
            label="Base image directory",
            placeholder=_first.base_dir_placeholder,
        )
        csv_path = gr.Textbox(
            label="CSV path (optional)",
            placeholder=_first.csv_placeholder,
        )
        extra_field = gr.Textbox(
            label=_first.extra_field_label,
            placeholder=_first.extra_placeholder,
            visible=bool(_first.extra_field_label),
        )
        extra_field2 = gr.Textbox(
            label=_first.extra_field2_label,
            placeholder=_first.extra_placeholder2,
            visible=bool(_first.extra_field2_label),
        )
        _dd_choices = list(_first.extra_dropdown_choices)
        extra_dropdown = gr.Dropdown(
            choices=_dd_choices,
            value=(_dd_choices[0] if _dd_choices else None),
            label=_first.extra_dropdown_label,
            visible=bool(_first.extra_dropdown_label),
        )
        cache_dir = gr.Textbox(label="Cache dir (leave blank to disable)")

        with gr.Group():
            gr.Markdown("**Output fields**")
            with gr.Row():
                want_cls = gr.Checkbox(label="Labels", value=True)
                want_mask = gr.Checkbox(label="Mask", value=False)
                want_report = gr.Checkbox(label="Report", value=False)
                want_bbox = gr.Checkbox(label="BBox", value=False)
                # Regression targets (continuous).  Datasets that don't
                # populate REG_COLS will silently ignore output_reg=True.
                want_reg = gr.Checkbox(label="Reg", value=False)

        with gr.Row():
            load_btn = gr.Button("Load Dataset", variant="primary", scale=3)
            stop_btn = gr.Button("Stop", variant="stop", scale=1, visible=False)
        load_status = gr.Textbox(
            label="Status", interactive=False, lines=2, max_lines=4
        )

    # Family change: repopulate variant dropdown choices + reset to first
    # variant + refresh placeholders for that variant in one shot.
    def _on_family_change(family_name):
        variants = families.get(family_name, [])
        if not variants:
            return (gr.update(),) * 6
        first = variants[0]
        ph = _update_extra_fields(first)
        return (
            gr.update(choices=variants, value=first, visible=len(variants) > 1),
            *ph,
        )

    family_dd.change(
        fn=_on_family_change,
        inputs=[family_dd],
        outputs=[
            variant_dd,
            base_dir,
            csv_path,
            extra_field,
            extra_field2,
            extra_dropdown,
        ],
        queue=False,
        show_progress="hidden",
    )
    # Variant change: refresh placeholders/labels for the newly selected variant.
    variant_dd.change(
        fn=_update_extra_fields,
        inputs=[variant_dd],
        outputs=[base_dir, csv_path, extra_field, extra_field2, extra_dropdown],
        queue=False,
        show_progress="hidden",
    )

    return (
        family_dd,
        variant_dd,
        base_dir,
        csv_path,
        extra_field,
        extra_field2,
        extra_dropdown,
        cache_dir,
        want_cls,
        want_mask,
        want_report,
        want_bbox,
        want_reg,
        load_btn,
        load_status,
        stop_btn,
    )


def _output_panel():
    """Render the shared gallery + metadata + code output panel."""
    gallery = gr.Gallery(label="③ Results", columns=4, object_fit="contain", height=600)
    with gr.Group():
        gr.Markdown("### Sample info")
        sample_info = gr.Textbox(label="Path", interactive=False)
        image_stats = gr.Textbox(label="Image size & intensity", interactive=False)
        with gr.Row():
            labels_out = gr.Textbox(label="Labels", interactive=False, scale=1)
            report_out = gr.Textbox(label="Report", interactive=False, scale=2, lines=3)
        qa_out = gr.Textbox(label="Q & A", interactive=False, lines=8, visible=True)
    with gr.Accordion(
        "Transform code snippet", open=False, elem_classes=["aug-accordion"]
    ):
        code_out = gr.Code(label="", language="python", interactive=False)
    return labels_out, report_out, qa_out, sample_info, image_stats, gallery, code_out


def build_dataset_tab(modality: str, families: dict[str, list[str]]) -> None:
    """Render one modality's full tab body — load panel, controls, runner, outputs.

    All datasets within a modality must currently share the same ``is_3d``
    dimensionality.  Mixed-dim modalities aren't supported yet: they'd need
    per-dataset control adaptation (3-D controls would appear/disappear as the
    dropdown changes) which is a bigger UI change than we want to take on here.
    """
    all_keys = [k for variants in families.values() for k in variants]
    is_3d_flags = {DATASET_REGISTRY[k].is_3d for k in all_keys}
    if len(is_3d_flags) != 1:
        raise ValueError(
            f"Modality {modality!r} mixes is_3d values ({is_3d_flags}) across "
            f"{all_keys}.  Split into separate modalities or add per-dataset "
            "adapter logic before enabling this mix."
        )
    is_3d = is_3d_flags.pop()

    state = gr.State(None)
    last_sample = gr.State(None)
    gallery_samples_state = gr.State([])  # cached post-transform samples for next-bbox
    bbox_idx_state = gr.State(0)  # current bbox index for cervical/lumbar
    _native_spacing_sink = gr.State(
        None
    )  # absorbs 3rd return value of load_dataset in 2-D tabs

    with gr.Row():
        with gr.Column(scale=1):
            (
                family_dd,
                variant_dd,
                base_dir,
                csv_path,
                extra_field,
                extra_field2,
                extra_dropdown,
                cache_dir,
                want_cls,
                want_mask,
                want_report,
                want_bbox,
                want_reg,
                load_btn,
                load_status,
                stop_btn,
            ) = _dataset_panel(families)

            # HU (Hounsfield Units) only make sense for CT — MRI intensities are
            # scanner/sequence-dependent, so hide the windowing controls and
            # default them off on MRI tabs.  Kept as real (hidden) components
            # so the run_3d wiring stays uniform across 3-D modalities.
            hu_applicable = is_3d and modality != "MRI"
            with gr.Group():
                gr.Markdown("## ② Preprocessing & Display")
                if is_3d:
                    img_size = gr.Slider(
                        112, 512, value=224, step=16, label="Volume size"
                    )
                    pad = gr.Checkbox(label="Pad to cube", value=True)
                    hu_window = gr.Checkbox(
                        label="HU windowing",
                        value=hu_applicable,
                        visible=hu_applicable,
                    )
                    with gr.Row(visible=hu_applicable):
                        hu_min = gr.Number(value=-1000, label="HU min")
                        hu_max = gr.Number(value=1000, label="HU max")
                    # Voxel spacing (mm).  On by default so anisotropic MRI
                    # (e.g. 0.5 x 0.5 x 4 mm lumbar spine) doesn't collapse
                    # along z under ResizeD(size_mode="longest").  Per-axis
                    # controls let users match the dataset's native anisotropy
                    # if needed.  Disable for raw-geometry inspection or when
                    # an upstream step already resampled.
                    gr.Markdown("**Voxel spacing (mm)**")
                    iso_spacing = gr.Checkbox(label="Resample", value=True)
                    # Native spacing readout — placed ABOVE the sliders so the
                    # user sees the dataset's starting point before they touch
                    # anything.  Populated by load_dataset.
                    native_spacing_md = gr.Markdown(
                        "Native spacing: _load a dataset to see_",
                        elem_classes=["native-spacing-readout"],
                    )
                    with gr.Row():
                        pixdim_x = gr.Slider(
                            0.25, 5.0, value=1.0, step=0.05, label="x (mm)"
                        )
                        pixdim_y = gr.Slider(
                            0.25, 5.0, value=1.0, step=0.05, label="y (mm)"
                        )
                        pixdim_z = gr.Slider(
                            0.25, 5.0, value=1.0, step=0.05, label="z (mm)"
                        )
                    # Preset shortcuts.  Without an explicit "Presets:" label
                    # the buttons read as orphaned text in the panel.  Variant
                    # matches "Run on same sample" (secondary) for visual
                    # consistency.
                    gr.Markdown("**Presets**")
                    with gr.Row():
                        preset_1mm_btn = gr.Button(
                            "1 mm isotropic", variant="secondary"
                        )
                        preset_match_btn = gr.Button(
                            "Match native", variant="secondary"
                        )
                else:
                    img_size = gr.Slider(
                        64, 1024, value=224, step=32, label="Image size"
                    )
                    pad = gr.Checkbox(label="Pad to square", value=True)
                    hu_window = hu_min = hu_max = None
                    iso_spacing = pixdim_x = pixdim_y = pixdim_z = None
                    native_spacing_md = None
                    preset_1mm_btn = preset_match_btn = None
                with gr.Row():
                    show_mask = gr.Checkbox(label="Mask overlay", value=True)
                    show_bbox = gr.Checkbox(label="BBox overlay", value=True)

        with gr.Column(scale=2):
            aug_inputs = aug_controls(is_3d=is_3d)

    with gr.Row():
        run_btn = gr.Button("▶  Run on random sample", variant="primary", size="lg")
        run_same_btn = gr.Button(
            "▶  Run on same sample",
            variant="secondary",
            size="lg",
            interactive=False,
        )
        # Cervical/lumbar only: cycle to the next bbox without re-running the
        # transform pipeline.  Hidden on 2-D tabs entirely; on 3-D tabs starts
        # hidden and is revealed only when a focus-first sample with bboxes
        # has been rendered.
        next_bbox_btn = gr.Button(
            "Next bbox →",
            variant="secondary",
            size="lg",
            interactive=False,
            visible=False,
        )
    if is_3d:
        gr.Markdown(
            "_Each panel: **axial | coronal | sagittal** mid-slices. "
            "Gallery order: **Original** → individual augmentations → **Compound** (when ≥2 enabled)_"
        )
    else:
        gr.Markdown(
            "_Gallery order: **Original** → individual augmentations → **Compound** (when ≥2 enabled)_"
        )

    labels_out, report_out, qa_out, sample_info, image_stats, gallery, code_out = (
        _output_panel()
    )

    # --- Event wiring ----------------------------------------------------
    load_btn.click(
        fn=lambda: (gr.update(visible=False), gr.update(visible=True)),
        outputs=[load_btn, stop_btn],
        queue=False,
    )
    load_event = load_btn.click(
        fn=load_dataset,
        inputs=[
            variant_dd,
            base_dir,
            csv_path,
            extra_field,
            extra_field2,
            extra_dropdown,
            cache_dir,
            want_cls,
            want_mask,
            want_report,
            want_bbox,
            want_reg,
        ],
        # load_dataset always returns 3 values.  3-D tabs route the third
        # (native-spacing string) into the markdown caption; 2-D tabs sink
        # it into a hidden gr.State to avoid a Gradio "too many return values"
        # warning that would print the full state dict to the terminal.
        outputs=(
            [state, load_status, native_spacing_md]
            if is_3d
            else [state, load_status, _native_spacing_sink]
        ),
        show_progress_on=[load_status],
    )
    load_event.then(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[load_btn, stop_btn],
        queue=False,
    )
    stop_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[load_btn, stop_btn],
        cancels=[load_event],
        queue=False,
    )

    # run_3d/run_3d_same accept the extra HU args; run_2d variants don't.
    run_fn = run_3d if is_3d else run_2d
    run_same_fn = run_3d_same if is_3d else run_2d_same
    extra_3d_inputs = (
        [hu_window, hu_min, hu_max, iso_spacing, pixdim_x, pixdim_y, pixdim_z]
        if is_3d
        else []
    )

    run_btn.click(
        fn=run_fn,
        inputs=[state, img_size, pad, show_mask, show_bbox]
        + extra_3d_inputs
        + list(aug_inputs),
        outputs=[
            gallery,
            sample_info,
            image_stats,
            labels_out,
            report_out,
            qa_out,
            code_out,
            last_sample,
            gallery_samples_state,
            run_same_btn,
        ],
    ).then(
        # Fresh sample → reset bbox cursor and re-evaluate next-bbox interactivity.
        fn=_next_btn_state_after_run,
        inputs=[state, gallery_samples_state],
        outputs=[bbox_idx_state, next_bbox_btn],
    )
    run_same_btn.click(
        fn=run_same_fn,
        inputs=[state, last_sample, img_size, pad, show_mask, show_bbox]
        + extra_3d_inputs
        + list(aug_inputs),
        outputs=[
            gallery,
            sample_info,
            image_stats,
            labels_out,
            report_out,
            qa_out,
            code_out,
            last_sample,
            gallery_samples_state,
        ],
    ).then(
        # Same-sample re-run keeps the bbox set; just re-check button interactivity.
        fn=_next_btn_interactive,
        inputs=[state, gallery_samples_state],
        outputs=[next_bbox_btn],
    )

    next_bbox_btn.click(
        fn=_on_next_bbox,
        inputs=[gallery_samples_state, bbox_idx_state, show_mask, show_bbox, state],
        outputs=[gallery, bbox_idx_state],
    )

    # Reset the bbox cursor when the user toggles bbox visibility off/on or
    # switches dataset variant — the next click should start from index 0.
    show_bbox.change(
        fn=lambda: 0,
        inputs=[],
        outputs=[bbox_idx_state],
        queue=False,
        show_progress="hidden",
    )
    variant_dd.change(
        fn=lambda: 0,
        inputs=[],
        outputs=[bbox_idx_state],
        queue=False,
        show_progress="hidden",
    )

    # ── 3-D-only event wiring: spacing presets + live re-run on slider release.
    # Per items 1, 2, 3 of docs/proposals/voxel_spacing_ui.md (Frank-approved).
    if is_3d:
        # Preset buttons just set the slider values.  Re-render only happens
        # when the user clicks "Run on same sample" — spacing edits don't
        # auto-fire the pipeline.
        def _preset_1mm():
            return gr.update(value=1.0), gr.update(value=1.0), gr.update(value=1.0)

        def _preset_match(s):
            if s and s.get("native_spacing"):
                nx, ny, nz = s["native_spacing"]
                return (
                    gr.update(value=round(nx, 2)),
                    gr.update(value=round(ny, 2)),
                    gr.update(value=round(nz, 2)),
                )
            return gr.update(), gr.update(), gr.update()

        preset_1mm_btn.click(fn=_preset_1mm, outputs=[pixdim_x, pixdim_y, pixdim_z])
        preset_match_btn.click(
            fn=_preset_match,
            inputs=[state],
            outputs=[pixdim_x, pixdim_y, pixdim_z],
        )


# ---------------------------------------------------------------------------
# Build UI
# ---------------------------------------------------------------------------

_CSS = """
.prose h1 { font-size: 2rem !important; font-weight: 700 !important; letter-spacing: -0.02em !important; }
.prose h2 { font-size: 1.35rem !important; font-weight: 600 !important; }
.prose h3 { font-size: 1.15rem !important; font-weight: 600 !important; }
.prose p  { font-size: 1.05rem !important; line-height: 1.6 !important; }
.prose em { color: #6b7280 !important; }
button.primary, button.secondary, button.stop { border-radius: 8px !important; }
.gr-group { overflow: visible !important; padding-bottom: 8px !important; }
.aug-accordion .label-wrap { background: #3f3f46; color: #ffffff !important; border-radius: 10px !important; padding: 6px 14px !important; }
.aug-accordion .label-wrap span { font-size: 1.05rem !important; font-weight: 600 !important; color: #ffffff !important; }
.aug-accordion > button { background: #3f3f46; color: #ffffff !important; border-radius: 10px !important; padding: 6px 14px !important; font-size: 1.05rem !important; font-weight: 600 !important; }
.aug-accordion > div:first-child { background: #3f3f46; color: #ffffff !important; border-radius: 10px !important; padding: 6px 14px !important; }
.native-spacing-readout p { 
    font-size: 0.95em !important; 
    padding: 6px 10px !important; 
    margin: 4px 0 !important; 
    border-radius: 4px !important; }
"""

with gr.Blocks(title="RadHarmony Visualizer", css=_CSS) as demo:
    gr.Markdown("# RadHarmony — Dataset & Transform Visualizer")
    gr.Markdown(
        "**Workflow:** &nbsp; ① Pick a dataset and click **Load** &nbsp;→&nbsp; "
        "② Adjust preprocessing & augmentations &nbsp;→&nbsp; "
        "Click **Run on random sample** to inspect each effect side-by-side."
    )

    # ── One tab per modality ─────────────────────────────────────────────────
    # SUPPORTED_MODALITIES defines the display order; modalities with no
    # registered datasets are skipped silently.
    for _modality in SUPPORTED_MODALITIES:
        _families = _MODALITY_FAMILIES.get(_modality, {})
        if not _families:
            continue
        with gr.Tab(_modality):
            build_dataset_tab(_modality, _families)


if __name__ == "__main__":
    import argparse as _ap
    _p = _ap.ArgumentParser()
    _p.add_argument("--server-port", type=int, default=7860)
    _args, _ = _p.parse_known_args()
    demo.launch(server_name="0.0.0.0", server_port=_args.server_port)
