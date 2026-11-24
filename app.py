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
That is all. The UI and loader update automatically.
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
    ChestXray14BboxDataset,
    ChestXDet10TrainDataset,
    Node21Dataset,
    Node21OriginalDataset,
    ChestXDet10TestDataset,
    ChestXray14TrainDataset,
    ChestXray14TestDataset,
    CheXpertTrainDataset,
    CheXpertValidDataset,
    CheXpertPlusTrainDataset,
    CheXpertPlusValidDataset,
    CheXphotoTrainDataset,
    CheXphotoValidDataset,
    CheXlocalizeDataset,
    CheXlocalizeValDataset,
    MIMICCXRTrainDataset,
    MIMICCXRValidateDataset,
    MIMICCXRTestDataset,
    MIMICCXRJPGTrainDataset,
    MIMICCXRJPGValidateDataset,
    MIMICCXRJPGTestCheXpertDataset,
    MIMICCXRJPGTestDataset,
    PadChestDataset,
    PadChestGRTrainDataset,
    PadChestGRValidationDataset,
    PadChestGRTestDataset,
    QaTaCOV19TrainDataset,
    QaTaCOV19TestDataset,
    MedSegCOVIDCTDataset,
    MedSegCOVIDCTValDataset,
    RadChestCTTrainDataset,
    RadChestCTValidDataset,
    RadChestCTTestDataset,
    ReXGradientTrainDataset,
    ReXGradientValidDataset,
    ReXGradientTestDataset,
    RANZCRClipTrainDataset,
    RANZCRClipTestDataset,
    RSNA2022CervicalSpineBboxDataset,
    RSNA2022CervicalSpineTrainDataset,
    RSNA2022CervicalSpineTestDataset,
    RSNA2024LumbarSpineTrainDataset,
    RSNA2024LumbarSpineTestDataset,
    RSNAAbdominalTrauma2023TrainDataset,
    RSNAAbdominalTrauma2023TestDataset,
    RSNABoneAgeTrainDataset,
    RSNABoneAgeValDataset,
    RSNABoneAgeTestDataset,
    RSNAPEDetectionTrainDataset,
    RSNAPEDetectionTestDataset,
    RSNAPneumoniaDataset,
    RSNAPneumoniaKaggleTrainDataset,
    RSNAPneumoniaKaggleTestDataset,
    SIIMACRPTXTrainDataset,
    SIIMACRPTXTestDataset,
    SIIMCOVID19TrainDataset,
    SIIMCOVID19TestDataset,
    CTRATETrainDataset,
    CTRATEValidDataset,
    TAIXRayTrainDataset,
    TAIXRayValDataset,
    TAIXRayTestDataset,
    TAIXRay512TrainDataset,
    TAIXRay512ValDataset,
    TAIXRay512TestDataset,
    MontgomeryCXRDataset,
    OpenICXRDataset,
    ShenzhenCXRDataset,
    VinDrCXRTestDataset,
    VinDrCXRTrainDataset,
    VinDrPCXRTrainDataset,
    VinDrPCXRTestDataset,
    EmoryCXRDataset,
    EmoryCHORUSDataset,
    MSCXRTrainDataset,
    MSCXRValDataset,
    MSCXRTestDataset,
    MSCXRTDataset,
    ChestImaGenomeGoldDataset,
    ChestImaGenomeSilverTrainDataset,
    ChestImaGenomeSilverValidDataset,
    ChestImaGenomeSilverTestDataset,
)
from radharmony.dataset.transforms import RadiologyTransform2D, RadiologyTransform3D
from radharmony.utils.data_utils import get_data_dict
from radharmony.utils.infer import infer_path as _infer


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------


#: Controlled vocabulary of imaging modalities.  Add new values here first so
#: the UI grouping and dataset registry stay in sync.  Keep the list short,
#: because a long list crowds the tab strip.
SUPPORTED_MODALITIES: tuple = (
    "CXR",  # chest X-ray
    "Radiograph",  # general radiography: hand, bone, extremity (anything not chest)
    "CT",  # computed tomography (3-D volumes)
    "CT (2D)",  # computed tomography served as 2-D slices (uses the 2-D pipeline)
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
            for UI grouping. It is independent of ``is_3d``, which drives the
            transform pipeline.  It often matches the dimension (CXR is 2-D,
            CT is 3-D) but not always: some MRI acquisitions are 2-D.
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
    # Optional dataset-specific dropdown (for example, a series-type filter).
    # Empty label hides the control.  Choices must include a "(all)" sentinel
    # if the dataset wants to support "no filter".
    extra_dropdown_label: str = ""
    extra_dropdown_choices: tuple = ()
    extra_dropdown_kwarg: str = ""  # kwarg name forwarded to the build function
    # Group-header sentinel. It shows in the dropdown as a separator.
    # Selecting it does nothing (load_dataset returns early).
    is_group_header: bool = False
    # The output_bbox default of the build function. When the user picks this
    # entry, the BBox checkbox takes this value (D-11).
    bbox_default: bool = False

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


def _build_chexphoto(cls, default_csv, base_dir, csv_path, cache_dir, **flags):
    csv = _infer(base_dir, default_csv, user_path=csv_path)
    return (
        cls(
            base_image_dir=base_dir,
            csv_path=csv or None,
            subset=flags.get("chexphoto_subset") or "all",
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            # CheXphoto shows photography artifacts, so keep uncertain rows.
            # drop_uncertain deletes every row with an uncertain label (14 labels).
            # It would remove 12,487 of the 32,521 training images.
            drop_uncertain=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chexphoto_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return _build_chexphoto(
        CheXphotoTrainDataset, "train.csv", base_dir, csv_path, cache_dir, **flags
    )


def _build_chexphoto_valid(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return _build_chexphoto(
        CheXphotoValidDataset, "valid.csv", base_dir, csv_path, cache_dir, **flags
    )


def _build_chexpert_plus_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_chexpert_plus_cls(
        CheXpertPlusTrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir,
        **flags)


def _build_chexpert_plus_valid(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_chexpert_plus_cls(
        CheXpertPlusValidDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir,
        **flags)


def _build_chexpert_plus_cls(
    cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv = _infer(
        base_dir,
        "df_chexpert_plus_240401.csv",
        user_path=csv_path,
    )
    label_json = _infer(
        base_dir,
        "impression_fixed.json",
        user_path=extra_field,
    )
    return (
        cls(
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
    return _build_chexlocalize_cls(
        CheXlocalizeDataset, "test", base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_chexlocalize_val(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_chexlocalize_cls(
        CheXlocalizeValDataset, "val", base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_chexlocalize_cls(cls, split, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    csv = _infer(base_dir, f"{split}_labels.csv", user_path=csv_path)
    if not csv:
        return None, f"{split}_labels.csv is required."
    mask_json = _infer(
        base_dir, f"gt_segmentations_{split}.json", f"../CheXlocalize/gt_segmentations_{split}.json",  # D-35
        user_path=extra_field,
    ) or None
    mask_dir = extra_field2 or None
    if flags.get("output_mask") and mask_dir is None:
        return (
            None,
            "Mask output dir is required when Mask is enabled for CheXlocalize.",
        )
    return (
        cls(
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


def _build_mimic_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_mimic_cls(MIMICCXRTrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_mimic_validate(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_mimic_cls(MIMICCXRValidateDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_mimic_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_mimic_cls(MIMICCXRTestDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_mimic_cls(cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
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
    # The split file is in the MIMIC-CXR-JPG release, next to the Label CSV (V-A-37).
    from radharmony.harmonizer.mimic_cxr.mimic_split import SPLIT_FILE, find_next_to
    if not find_next_to(SPLIT_FILE, csv, label):
        return (
            None,
            f"{SPLIT_FILE} is not next to the Metadata CSV or the Label CSV. "
            "Fill the Label CSV box with mimic-cxr-2.0.0-chexpert.csv from the "
            "MIMIC-CXR-JPG folder, which has this file.",
        )
    return (
        cls(
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


def _build_mimic_jpg_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_mimic_jpg_cls(MIMICCXRJPGTrainDataset, base_dir, csv_path, extra_field, cache_dir, **flags)


def _build_mimic_jpg_validate(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_mimic_jpg_cls(MIMICCXRJPGValidateDataset, base_dir, csv_path, extra_field, cache_dir, **flags)


def _build_mimic_jpg_test_chexpert(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    # The test split of the split file, with CheXpert-labeler labels (D-09, A-17).
    return _build_mimic_jpg_cls(MIMICCXRJPGTestCheXpertDataset, base_dir, csv_path, extra_field, cache_dir, **flags)


def _build_mimic_jpg_cls(cls, base_dir, csv_path, extra_field, cache_dir, **flags):
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
        cls(
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


def _build_ctrate_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_ctrate_cls(CTRATETrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_ctrate_valid(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_ctrate_cls(CTRATEValidDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_ctrate_cls(cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    # The class finds the label and metadata tables of its split: the split
    # class, or the image folder name (train_* or valid_*) for CTRATEDataset.
    return (
        cls(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            view_position_csv_path=extra_field or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=flags.get("output_report", False),
            output_bbox=flags.get("output_bbox", False),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_radchestct_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_radchestct_cls(
        RadChestCTTrainDataset, ["imgtrain_Abnormality_and_Location_Labels.csv"],
        base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_radchestct_valid(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_radchestct_cls(
        RadChestCTValidDataset, ["imgvalid_Abnormality_and_Location_Labels.csv"],
        base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_radchestct_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_radchestct_cls(
        RadChestCTTestDataset, ["imgtest_Abnormality_and_Location_Labels.csv"],
        base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_radchestct_cls(
    cls, names, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    # The tables are in tables/ of the folder above the image folder (D-35).
    csv_path = _infer(
        base_dir, "CT_Scan_Metadata_Complete_35747.csv", "tables/CT_Scan_Metadata_Complete_35747.csv",
        user_path=csv_path,
    )
    if not csv_path:
        return None, "Metadata CSV (CT_Scan_Metadata_Complete_35747.csv) is required."
    extra_field = _infer(base_dir, *names, *(f"tables/{n}" for n in names), user_path=extra_field)
    if not extra_field:
        return (
            None,
            "Label CSV (imgtrain/imgvalid/imgtest_Abnormality_and_Location_Labels.csv) is required.",
        )
    bbox_csv = (
        _infer(
            base_dir,
            "Extrema_35747.csv",
            "tables/Extrema_35747.csv",
            user_path=extra_field2,
        )
        or None
    )
    return (
        cls(
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
    return (
        ChestXray14TrainDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
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
    return (
        ChestXray14TestDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=False,
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_node21(cls, base_dir, csv_path, cache_dir, **flags):
    return (
        cls(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            # Nodule boxes are the reason NODE21 exists, so request them unless
            # the caller has explicitly said otherwise.
            output_bbox=flags.get("output_bbox", True),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_node21_processed(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return _build_node21(Node21Dataset, base_dir, csv_path, cache_dir, **flags)


def _build_node21_original(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return _build_node21(
        Node21OriginalDataset, base_dir, csv_path, cache_dir, **flags)


def _build_chestxdet10(cls, split, base_dir, csv_path, cache_dir, **flags):
    csv_path = _infer(base_dir, f"{split}.json", user_path=csv_path)
    if not csv_path:
        return None, f"{split}.json is required (ships with the Deepwise release)."
    return (
        cls(
            base_image_dir=base_dir,
            csv_path=csv_path,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            # ChestX-Det10 has neither masks nor reports, so the dataset class
            # takes no output_mask / output_report kwargs.  Boxes are the whole
            # point of it, so request them unless told otherwise.
            output_bbox=flags.get("output_bbox", True),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chestxdet10_train(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return _build_chestxdet10(
        ChestXDet10TrainDataset, "train", base_dir, csv_path, cache_dir, **flags
    )


def _build_chestxdet10_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return _build_chestxdet10(
        ChestXDet10TestDataset, "test", base_dir, csv_path, cache_dir, **flags
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


def _build_emory_cxr(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    """Build the Emory CXR v2 dataset (PNG tree).

    ``extra_field`` is the finding-label CSV and ``extra_field2`` is the
    report CSV. Both are optional. The loader reads them only when the
    matching output flag is set.

    ``csv_path`` names one of the dated metadata manifests.  They are not
    interchangeable -- successive versions filter the cohort -- so the
    error message says so rather than quietly inferring one.
    """
    csv_path = _infer(
        base_dir,
        "EmoryCXR_v2_Metadata_08152025.csv",
        "EmoryCXR_v2_Metadata_08142025.csv",
        "EmoryCXR_v2_Metadata_08112025.csv",
        "EmoryCXR_v2_Metadata_10172024.csv",
        # The same names in TABLES/ of the folder above the image folder (D-35).
        "TABLES/EmoryCXR_v2_Metadata_08152025.csv",
        "TABLES/EmoryCXR_v2_Metadata_08142025.csv",
        "TABLES/EmoryCXR_v2_Metadata_08112025.csv",
        "TABLES/EmoryCXR_v2_Metadata_10172024.csv",
        user_path=csv_path,
    )
    if not csv_path:
        return None, (
            "EmoryCXR_v2_Metadata_<date>.csv is required. The dated "
            "manifests are not interchangeable; pass one explicitly."
        )
    return (
        EmoryCXRDataset(
            base_image_dir=base_dir,
            csv_path=csv_path,
            label_csv_path=extra_field or None,
            report_csv_path=extra_field2 or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=False,
            output_report=flags.get("output_report", False),
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_emory_chorus(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    """Build the Emory CHORUS X-ray subset (DICOM tree, chest only, image only).

    ``csv_path`` is the manifest CSV (a scan cache). If it does not exist, the
    first build scans the DICOM tree and writes it there.
    """
    if not base_dir:
        return None, "Base dir (the CHORUS images_dicom_xray folder) is required."
    return (
        EmoryCHORUSDataset(
            base_image_dir=base_dir,
            manifest_csv_path=csv_path or None,
            cache_dir=cache_dir or None,
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


def _build_padchest_gr_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_padchest_gr_cls(
        PadChestGRTrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_padchest_gr_validation(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_padchest_gr_cls(
        PadChestGRValidationDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir,
        **flags)


def _build_padchest_gr_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_padchest_gr_cls(
        PadChestGRTestDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_padchest_gr_cls(
    cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    return (
        cls(
            base_image_dir=base_dir,
            csv_path=csv_path or None,
            padchest_csv_path=extra_field or None,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_report=flags.get("output_report", False),
            # The boxes are what PadChest-GR adds to PadChest, so request them
            # unless the caller has explicitly said otherwise.
            output_bbox=flags.get("output_bbox", True),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_rexgradient(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, split, dataset_cls, json_name, **flags
):
    """Common build path for the three ReXGradient splits.

    The JSON metadata is the source of truth. The per-study CSV next to it
    cannot give per-image rows. A CSV path gives an error, so the user
    does not get the wrong data shape.
    """
    csv_path = _infer(base_dir, json_name, f"download/metadata/{json_name}", user_path=csv_path)  # D-35
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
        "annotations/image_labels_train.csv",  # in the folder above (D-35)
        user_path=csv_path,
    )
    if not csv_path:
        return None, "Label CSV (image_labels_train.csv) is required."
    bbox_csv = (
        _infer(
            base_dir,
            "annotations_train.csv",
            "annotations/annotations_train.csv",
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
    # base_dir is the MIMIC-CXR DICOM files/ root. extra_field is the Chest ImaGenome root.
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


def _build_chest_imagenome_silver_split(cls, base_dir, extra_field, cache_dir, **flags):
    if not extra_field:
        return None, "Annotation dir (Chest ImaGenome release root) is required."
    return (
        cls(
            base_image_dir=base_dir,
            annotation_dir=extra_field,
            cache_dir=cache_dir or None,
            output_bbox=flags.get("output_bbox", True),
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_chest_imagenome_silver_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_chest_imagenome_silver_split(
        ChestImaGenomeSilverTrainDataset, base_dir, extra_field, cache_dir, **flags)


def _build_chest_imagenome_silver_valid(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_chest_imagenome_silver_split(
        ChestImaGenomeSilverValidDataset, base_dir, extra_field, cache_dir, **flags)


def _build_chest_imagenome_silver_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_chest_imagenome_silver_split(
        ChestImaGenomeSilverTestDataset, base_dir, extra_field, cache_dir, **flags)


def _build_vindr_cxr_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    csv_path = _infer(
        base_dir,
        "image_labels_test.csv",
        "annotations/image_labels_test.csv",  # in the folder above (D-35)
        user_path=csv_path,
    )
    if not csv_path:
        return None, "Label CSV (image_labels_test.csv) is required."
    bbox_csv = (
        _infer(
            base_dir,
            "annotations_test.csv",
            "annotations/annotations_test.csv",
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


def _build_vindr_pcxr_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_vindr_pcxr_cls(
        VinDrPCXRTrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_vindr_pcxr_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_vindr_pcxr_cls(
        VinDrPCXRTestDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_vindr_pcxr_cls(
    cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, "Dataset root directory (containing train/ and test/) is required."
    return (
        cls(
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


def _build_rsna_bone_age_test(
    base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags
):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    return (
        RSNABoneAgeTestDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
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
    # Expand ~ in both paths. Gradio passes the raw string.
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


def _build_taix_ray_512_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_taix_ray(TAIXRay512TrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_taix_ray_512_val(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_taix_ray(TAIXRay512ValDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_taix_ray_512_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_taix_ray(TAIXRay512TestDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_taix_ray_original_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_taix_ray(TAIXRayTrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_taix_ray_original_val(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_taix_ray(TAIXRayValDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_taix_ray_original_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_taix_ray(TAIXRayTestDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


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
    from radharmony.harmonizer.openi_cxr import OpenICXRHarmonizer

    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, "Dataset root directory (containing ecgen-radiology/ and DICOM folders) is required."

    try:
        h = OpenICXRHarmonizer(base_dir=base_dir)
        df = h.harmonize()
    except Exception as e:
        return None, f"Harmonization error: {e}"

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
        return None, "The CXR_png/ folder (base_image_dir) is required."
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
        return None, "The CXR_png/ folder (base_image_dir) is required."
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


def _build_qata_cov19_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_qata_cov19_cls(
        QaTaCOV19TrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_qata_cov19_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_qata_cov19_cls(
        QaTaCOV19TestDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_qata_cov19_cls(
    cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, (
            "A QaTa-COV19-v2 split's Images/ directory is required "
            "(e.g. .../QaTa-COV19-v2/Train Set/Images/)."
        )
    # Masks are ready-made PNGs in the sibling Ground-truths/ dir.
    # The second field is only for trees that were moved.
    masks_dir = os.path.expanduser(extra_field) if extra_field else None
    return (
        cls(
            base_image_dir=base_dir,
            masks_dir=masks_dir,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=False,
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_medseg_covid_ct(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir:
        return None, ("Dataset folder is required: the one holding tr_im.nii.gz "
                      "(dataset1) or rp_im/ (dataset2).")
    slice_dir = os.path.expanduser(extra_field) if extra_field else None
    if not slice_dir:
        return None, ("Slice output dir is required. MedSeg ships CT volumes; one PNG "
                      "per slice is written there once before MONAI can load them.")
    subset = (extra_field2 or "dataset1").strip() or "dataset1"
    if subset not in ("dataset1", "dataset2"):
        return None, "Subset must be 'dataset1' or 'dataset2'."
    return (
        MedSegCOVIDCTDataset(
            base_image_dir=base_dir,
            slice_output_dir=slice_dir,
            subset=subset,
            cache_dir=cache_dir or None,
            output_cls=flags.get("output_cls", False),
            output_mask=flags.get("output_mask", False),
            output_report=False,
            output_bbox=False,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_medseg_covid_ct_val(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    if not base_dir or not extra_field:
        return None, "The folder with val_im.nii.gz and the slice output dir are required."
    return (
        MedSegCOVIDCTValDataset(
            base_image_dir=base_dir,
            slice_output_dir=os.path.expanduser(extra_field),
            cache_dir=cache_dir or None,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_ms_cxr_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_ms_cxr_cls(MSCXRTrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_ms_cxr_val(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_ms_cxr_cls(MSCXRValDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_ms_cxr_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_ms_cxr_cls(MSCXRTestDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_ms_cxr_cls(
    cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    csv_path = os.path.expanduser(csv_path) if csv_path else csv_path
    if not base_dir:
        return None, "MIMIC-CXR files/ directory (JPG or DICOM tree) is required."
    if not csv_path:
        return None, "CSV path (MS-CXR local alignment CSV) is required."
    return (
        cls(
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
        return None, "MIMIC-CXR-JPG files/ folder is required."
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


def _build_ranzcr_clip_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    return _build_ranzcr_clip_cls(
        RANZCRClipTrainDataset, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags)


def _build_ranzcr_clip_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    base_dir = os.path.expanduser(base_dir) if base_dir else base_dir
    return (
        RANZCRClipTestDataset(
            base_image_dir=base_dir,
            cache_dir=cache_dir or None,
            dtype=_APP_DTYPE,
        ),
        None,
    )


def _build_ranzcr_clip_cls(cls, base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
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
        cls(
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


def _build_roco_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.harmonizer.roco import ROCOTrainHarmonizer
    return _build_roco_cls(ROCOTrainHarmonizer, base_dir)


def _build_roco_validation(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.harmonizer.roco import ROCOValidationHarmonizer
    return _build_roco_cls(ROCOValidationHarmonizer, base_dir)


def _build_roco_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.harmonizer.roco import ROCOTestHarmonizer
    return _build_roco_cls(ROCOTestHarmonizer, base_dir)


def _build_roco_cls(harmonizer_cls, base_dir):
    """Harmonize ROCO with a split harmonizer ``harmonizer_cls`` and wrap the table for the app.

    base_dir → roco-dataset root directory (contains data/)
    """
    from tqdm import tqdm

    base_dir_exp = os.path.expanduser(base_dir) if base_dir else None
    if not base_dir_exp or not os.path.isdir(base_dir_exp):
        return None, "ROCO base_dir (roco-dataset root with data/) not found."

    data_dir = os.path.join(base_dir_exp, "data")
    if not os.path.isdir(data_dir):
        return None, f"Expected data/ subdirectory not found in {base_dir_exp}."

    try:
        h = harmonizer_cls(
            base_dir=base_dir_exp,
            splits=None,  # a split class keeps its own split
            radiology_only=True,
        )
        df = h.harmonize()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"

    # The app shows only a fixed key set, so the caption goes into qa_text (V-A-32).
    df = df.copy()
    df["qa_text"] = [
        f"Caption: {a}" + (f"\nKeywords: {k}" if k else "")
        for a, k in zip(df["answer"], df["keywords"].fillna(""), strict=True)
    ]

    class _ROCOAdapter:
        LABEL_COLS = []
        REG_COLS   = []
        _cols = {"qa_text": "qa_text"}

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


def _build_rocov2_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.harmonizer.rocov2 import ROCOv2TrainHarmonizer
    return _build_rocov2_cls(ROCOv2TrainHarmonizer, base_dir, **flags)


def _build_rocov2_validation(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.harmonizer.rocov2 import ROCOv2ValidationHarmonizer
    return _build_rocov2_cls(ROCOv2ValidationHarmonizer, base_dir, **flags)


def _build_rocov2_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.harmonizer.rocov2 import ROCOv2TestHarmonizer
    return _build_rocov2_cls(ROCOv2TestHarmonizer, base_dir, **flags)


def _build_rocov2_cls(harmonizer_cls, base_dir, **flags):
    """Harmonize ROCOv2 with ``harmonizer_cls`` and wrap the table for the app.

    ROCOv2 has 79,789 radiology figure-caption pairs from PubMed Central. Its modality
    comes from the human-curated UMLS concepts (Zenodo <split>_concepts_manual.csv),
    named with cui_mapping.csv, so it does not depend on the caption words.

    base_dir → ROCOv2 root: hf/data/*.parquet plus the Zenodo CSVs, and images/ that
               scripts/materialize_rocov2.py writes
    flags    → ``rocov2_modality`` from the dropdown, and the output flags
    """
    from tqdm import tqdm

    bd = os.path.expanduser(base_dir) if base_dir else None
    if not bd or not os.path.isdir(bd):
        return None, "ROCOv2 base_dir not found."
    if not os.path.isdir(os.path.join(bd, "hf", "data")):
        return None, f"Expected hf/data/ under {bd}. Download eltorio/ROCOv2-radiology first."
    if not os.path.isdir(os.path.join(bd, "images")):
        return None, ("No images/ directory. Run "
                      "scripts/materialize_rocov2.py --base-dir <root> first: the HF "
                      "release stores image bytes inside parquet, not as files.")

    # The dropdown plumbing (see load_dataset) ALREADY maps "(all)"/"" -> None and
    # always sets the key, so pop()'s default never fires. Treat None as "no filter";
    # passing [None] through reaches the harmonizer as an unknown modality.
    mod = flags.pop("rocov2_modality", None) or "(all)"
    try:
        h = harmonizer_cls(
            base_dir=bd, splits=None,  # a split class keeps its own split
            modalities=None if mod in ("", "(all)") else [mod],
            require_modality=(mod not in ("", "(all)")),
        )
        df = h.harmonize()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"
    if df.empty:
        return None, f"No ROCOv2 rows for splits={h.splits} modality={mod}."

    # The app reads images from base_dir + image_path (see get_data_dict in load_dataset).
    # The harmonizer gives paths relative to <base_dir>/images.
    # Add the prefix here, and make verify_images use the same join.
    # Before, verify_images looked in <base_dir>/images but the loader looked in
    # <base_dir>, so the check passed and only LoadImaged failed.
    df = df.copy()
    df["image_path"] = "images/" + df["image_path"].astype(str)

    # The Sample-info panel only renders a FIXED set of keys -- see the filter in
    # load_dataset ("cls", "mask", "report", "qa_text", "bbox", "bbox_labels", "reg")
    # and the readers in _run(). Custom keys like "caption"/"cuis" are silently
    # dropped, which is why the panel came back blank. Fold everything worth reading
    # into qa_text, the box meant for VQA-schema datasets.
    def _rocov2_qa(row):
        lines = []
        cap = row.get("answer")
        if isinstance(cap, str) and cap.strip():
            lines += ["── Caption ──", cap.strip(), ""]
        mod, src = row.get("modality"), row.get("modality_source")
        # Pandas stores an unset modality as NaN, and `if NaN:` is true.
        # Without this check, the 105 unmapped test rows showed "Modality: nan".
        # `x != x` is the NaN test.
        if mod is None or mod != mod or str(mod).strip().lower() in ("", "nan", "none"):  # noqa: PLR0124
            mod = None
        n_mod = row.get("n_modality_cuis")
        n_mod = int(n_mod) if n_mod is not None and n_mod == n_mod else None  # noqa: PLR0124
        if mod:
            lines.append(f"Modality: {mod}" + (f"  (concepts: {src})" if src else ""))
        elif n_mod and n_mod > 1:
            # Deliberately left unset by the harmonizer: an image tagged both CT and
            # PET is genuinely both, and picking one would manufacture a label.
            lines.append(f"Modality: unset — {n_mod} different modalities in the concepts")
        else:
            lines.append("Modality: unset — no modality concept on this image")
        cuis = row.get("cuis")
        if isinstance(cuis, (list, tuple)) and len(cuis):
            lines.append(f"CUIs ({len(cuis)}): " + ", ".join(map(str, cuis[:25]))
                         + (" …" if len(cuis) > 25 else ""))
        if row.get("split"):
            lines.append(f"Split: {row['split']}")
        return "\n".join(lines).strip()

    df["qa_text"] = df.apply(_rocov2_qa, axis=1)

    class _ROCOv2Adapter:
        LABEL_COLS, REG_COLS = [], []
        _cols = {"qa_text": "qa_text"}   # caption + modality + CUIs -> Q&A box

        def __init__(self, d):
            self._df = d

        def get_harmonized_df(self):
            return self._df

        def verify_images(self, drop_missing=True):
            tqdm.pandas(desc="Verifying ROCOv2 images")
            mask = self._df["image_path"].progress_apply(
                lambda x: os.path.isfile(os.path.join(bd, x)))
            missing = self._df[~mask].copy()
            if not missing.empty and drop_missing:
                self._df = self._df[mask].reset_index(drop=True)
            return missing

    return _ROCOv2Adapter(df), None


def _build_mrrate(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    """MR-RATE: 705,254 brain/spine MRI volumes, 98,334 studies, with reports.

    3-D NIfTI, one row per series. Every study has a structured report and 37
    binary pathology labels. So you can check text and labels against the image.

    text field    -> splits filter (default: test)
    dropdown      -> sequence modality (T1w / T2w / FLAIR / SWI / MRA)

    The app shows only studies that are on disk. The HF repo is 8.11 TB and
    every study is one zip. Pull a cohort with scripts/materialize_mrrate.py first.
    """
    from radharmony.harmonizer.mrrate import MRRATEHarmonizer

    bd = os.path.expanduser((base_dir or "").strip())
    if not bd or not os.path.isdir(bd):
        return None, "MR-RATE base_dir not found."
    if not os.path.isdir(os.path.join(bd, "meta")):
        return None, (f"Expected meta/ under {bd}. Download the metadata first:\n"
                      "  hf download Forithmus/MR-RATE "
                      "--repo-type dataset --include '*.csv' --local-dir <base>/meta")

    splits = [s.strip() for s in (extra_field or "test").split(",") if s.strip()]
    mod = flags.pop("mrrate_modality", None) or "(all)"
    try:
        h = MRRATEHarmonizer(
            base_dir=bd, splits=splits,
            modalities=None if mod in ("", "(all)") else [mod],
            require_images=True,          # only offer volumes actually on disk
        )
        df = h.harmonize()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"
    if df.empty:
        return None, (f"No materialised MR-RATE volumes for splits={splits} "
                      f"modality={mod}. Pull some with:\n"
                      "  python scripts/materialize_mrrate.py --base-dir <base> "
                      "--split test --limit 20 --unpack")

    # Fold report + provenance into qa_text: the Sample-info panel renders only a
    # fixed key set, so bespoke columns never reach it (see the ROCOv2 note).
    def _qa(row):
        lines = [f"Study {row['study_id']}   ·   patient {row['patient_id']}",
                 f"Series: {row.get('series_id')}   {row.get('classified_modality')}"
                 f"   {row.get('acquisition_plane')}",
                 f"Sequence family: {row.get('sequence_family')}",
                 f"Split: {row.get('split')}   ·   centre series: "
                 f"{bool(row.get('is_center_modality'))}"]
        pos = [c.replace('_', ' ') for c in LBL if row.get(c) == 1]
        lines += ["", f"Pathology labels positive ({len(pos)} of 37): "
                      + (", ".join(pos) if pos else "none")]
        return "\n".join(lines)

    LBL = [c.lower().replace(" ", "_") for c in
           __import__("radharmony.harmonizer.mrrate", fromlist=["x"]).PATHOLOGY_LABELS]
    LBL = [c for c in LBL if c in df.columns]
    df = df.copy()
    df["qa_text"] = df.apply(_qa, axis=1)

    class _MRRATEAdapter:
        LABEL_COLS, REG_COLS = LBL, []
        _cols = {"report": "report", "qa_text": "qa_text", "mask": "mask_path"}

        def __init__(self, d):
            self._df = d

        def get_harmonized_df(self):
            return self._df

        def verify_images(self, drop_missing=True):
            # require_images=True already filtered, so this is a cheap re-check.
            mask = self._df["image_path"].apply(
                lambda x: os.path.isfile(os.path.join(bd, x)))
            missing = self._df[~mask].copy()
            if not missing.empty and drop_missing:
                self._df = self._df[mask].reset_index(drop=True)
            return missing

    return _MRRATEAdapter(df), None


def _build_medpmc(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    """MedPMC: 11.1M medical image-text pairs curated from PubMed Central.

    IMPORTANT: MedPMC ships NO specialty or modality field. The tags used here are
    RadHarmony's own, derived from caption text and optionally cross-checked against
    a zero-shot pixel tagger. Every row carries tag_method so they are never mistaken
    for author-provided labels.

    The dropdown selects a precision level. "high-precision radiology" needs both
    taggers to agree on specialty and modality, and the pixel tagger to be sure
    (p>=0.9). It has 648,694 images. In an expert review, 12 of 12 were correct
    (specialty and modality). This gives at least 74% precision at 95% confidence.
    It leaves out many valid images, so do not use it to estimate prevalence.

    base_dir     → MedPMC root containing webdataset/ and tags/
    extra_field  → modality filter (ct, mri, xray, ultrasound, angiography, ...)
    """
    from radharmony.harmonizer.medpmc import MedPMCHarmonizer
    from tqdm import tqdm

    bd = os.path.expanduser(base_dir) if base_dir else None
    if not bd or not os.path.isdir(bd):
        return None, "MedPMC base_dir not found."
    if not os.path.isfile(os.path.join(bd, "tags", "medpmc_tags.parquet")):
        return None, ("No tags/medpmc_tags.parquet. MedPMC ships no specialty field; "
                      "build the tag table with scripts/build_medpmc_tags.py first.")

    # The dropdown can give None, as for ROCOv2. Here this fails without an error.
    # A None preset matches no branch below, so no tag filter applies.
    # Then the whole 11M-row corpus loads, not the high-precision subset.
    preset = flags.pop("medpmc_preset", None) or "high-precision radiology"
    kw = dict(base_dir=bd, subsets=["singlepanel", "subfigure"])
    if preset == "high-precision radiology":
        kw.update(radiology_only=True, require_tagger_agreement=True, min_clip_prob=0.9)
    elif preset == "radiology (caption tagger only)":
        kw.update(radiology_only=True)
    elif preset == "radiology (both taggers agree)":
        kw.update(radiology_only=True, require_tagger_agreement=True)
    elif preset == "human-verified only":
        kw.update(radiology_only=False)   # the human verdict decides, not a tagger
    if extra_field and extra_field.strip():
        # Accept either spelling: "CT,MR" and "ct,mri" mean the same.
        # The table in medpmc/vocab.py is the single source of truth.
        from radharmony.harmonizer.medpmc.vocab import normalize as _norm_mod
        wanted = [_norm_mod(m) for m in extra_field.split(",") if m.strip()]
        bad = [m for m, n in zip(extra_field.split(","), wanted, strict=False) if n is None]
        if bad:
            return None, (f"Unknown modality {', '.join(b.strip() for b in bad)}. "
                          "Valid: CT MR XR US XA PT/NM MG (or ct mri xray "
                          "ultrasound angiography pet_spect mammography).")
        kw["modalities"] = wanted

    try:
        df = MedPMCHarmonizer(**kw).harmonize()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"
    if df.empty:
        return None, f"No MedPMC rows for preset={preset!r} modality={extra_field!r}."

    # Resolved labels: one modality per image plus WHERE it came from
    # (human > clip > figure_rollup > caption). Built by
    # scripts/build_medpmc_label_provenance.py. Optional -- the browser still works
    # without it, just with no provenance column.
    rp = os.path.join(bd, "tags", "medpmc_labels_resolved.parquet")
    if os.path.isfile(rp):
        import pandas as pd          # app.py has no module-level pandas import
        res = pd.read_parquet(rp, columns=["image_id", "modality_final",
                                           "modality_display", "modality_source",
                                           "specialty_final", "human_verified",
                                           "human_annotator", "sources_disagree"])
        df = df.merge(res, on="image_id", how="left")
        if preset == "human-verified only":
            df = df[df["human_verified"].fillna(False)]
    elif preset == "human-verified only":
        return None, ("No resolved-label table at tags/medpmc_labels_resolved.parquet. "
                      "Build it with scripts/build_medpmc_label_provenance.py.")

    # Same base_dir-vs-base_dir/images mismatch as ROCOv2; see the note there.
    df = df.copy()
    df["image_path"] = "images/" + df["image_path"].astype(str)

    # The panel has fixed keys, as for ROCOv2 (see the note there).
    # "caption" and "specialty" never reached the panel.
    # Put the caption and the source of our tags into qa_text.
    # The tags are ours, not from the dataset, so reviewers need their source.
    def _medpmc_qa(row):
        lines = []
        cap = row.get("answer")
        if isinstance(cap, str) and cap.strip():
            lines += ["── Caption ──", cap.strip(), ""]
        lines.append("── Derived tags (OURS, not shipped with MedPMC) ──")
        disp = row.get("modality_display")
        msrc = row.get("modality_source")
        if disp:
            lines.append(f"MODALITY: {disp}"
                         + (f"   (decided by: {msrc})" if msrc else ""))
        if row.get("human_verified"):
            lines.append(f"** HUMAN-VERIFIED ** by {row.get('human_annotator')}"
                         "  — a person judged this exact image")
        if row.get("sources_disagree"):
            lines.append("!! caption and CLIP taggers DISAGREE on this image")
        lines.append(f"specialty: {row.get('specialty')}   raw modality: {row.get('modality')}")
        lines.append(f"caption tagger: {row.get('tag_method')} "
                     f"(confidence {row.get('tag_confidence')}, source {row.get('tag_source')})")
        clip_p = row.get("clip_prob")
        lines.append(f"CLIP tagger: {row.get('clip_specialty')} / {row.get('clip_modality')}"
                     + (f"  p={float(clip_p):.3f}" if clip_p is not None and clip_p == clip_p else ""))  # noqa: PLR0124
        ident = [f"{k}={row.get(k)}" for k in ("pmcid", "image_id", "figure_label")
                 if row.get(k) is not None]
        if ident:
            lines += ["", "  ".join(ident)]
        return "\n".join(lines).strip()

    df["qa_text"] = df.apply(_medpmc_qa, axis=1)

    class _MedPMCAdapter:
        LABEL_COLS, REG_COLS = [], []
        _cols = {"qa_text": "qa_text"}   # caption + our tag provenance -> Q&A box

        def __init__(self, d):
            self._df = d

        def get_harmonized_df(self):
            return self._df

        def verify_images(self, drop_missing=True):
            # Images are inside 715 webdataset tars. Only a subset is on disk.
            # Extract what you need with scripts/materialize_medpmc_subset.py.
            tqdm.pandas(desc="Verifying MedPMC images")
            mask = self._df["image_path"].progress_apply(
                lambda x: os.path.isfile(os.path.join(bd, x)))
            missing = self._df[~mask].copy()
            if not missing.empty and drop_missing:
                self._df = self._df[mask].reset_index(drop=True)
            return missing

    return _MedPMCAdapter(df), None


def _gemex_box(loc, width: int, height: int) -> list:
    """Return a GEMeX ``[x1, y1, x2, y2]`` box as ``[y_min, y_max, x_min, x_max]`` fractions of the image (D-07).

    The box is in the 224 x 224 Chest ImaGenome space. That space resizes the image so that its long side
    is 224, then pads it to a square. The function undoes both steps and clips to [0, 1] (V-A-44).
    """
    side = 224.0
    scale = max(width, height) / side
    pad_x, pad_y = (side - width / scale) / 2, (side - height / scale) / 2
    x1, y1, x2, y2 = loc

    def frac(v, pad, size):
        return round(min(max((v - pad) * scale / size, 0.0), 1.0), 6)

    return [frac(y1, pad_y, height), frac(y2, pad_y, height), frac(x1, pad_x, width), frac(x2, pad_x, width)]


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

    # --- The Q&A text of each row: its own question and answer only (D-11, Q-3) ---
    _SUBTYPE_LABELS = {
        "closed_ended":  "Closed Ended (Yes / No)",
        "open_ended":    "Open Ended",
        "single_choice": "Single Choice",
        "multi_choice":  "Multi Choice",
    }

    def _format_qa(row):
        lines = [f"── {_SUBTYPE_LABELS.get(row['question_subtype'], row['question_subtype'])} ──",
                 f"Q ({row['question_type']}): {row['question']}"]
        choices = row.get("choices")
        if isinstance(choices, (list, tuple)) and choices:
            answer_letters = {a.strip() for a in str(row["answer"]).split(",")}
            lines.append("   " + "   ".join(
                f"{c} ✓" if c.split(":")[0].strip() in answer_letters else c for c in choices))
        lines.append(f"   → {row['answer']}")
        return "\n".join(lines)

    df["qa_text"] = [_format_qa(row) for _, row in df.iterrows()]

    # --- Bounding boxes: visual_locations are [x1,y1,x2,y2] in the 224 x 224 Chest ImaGenome space.
    # So each box needs the size of its image (_gemex_box). ---
    _sizes = {}

    def _image_size(rel):
        if rel not in _sizes:
            path = os.path.join(base_dir_exp or "", rel)
            _sizes[rel] = Image.open(path).size if os.path.isfile(path) else None  # header only
        return _sizes[rel]

    def _norm_bboxes(row):
        locs = [loc for loc in (row.get("visual_locations") or []) if len(loc) == 4]
        size = _image_size(row["image_path"]) if locs else None
        if not size:
            return None
        return _json.dumps([_gemex_box(loc, *size) for loc in locs])

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
            "qa_text":     "qa_text",            # the Q&A of this row → Q&A box
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


class _TableAdapter:
    """A ready table with the methods that ``load_dataset`` calls.

    Args:
        df: Table with an ``image_path`` column.
        base_dir: Folder that a relative ``image_path`` is relative to.
        name: Dataset name for the progress bar.
        cols: Sample key -> table column, as in ``get_data_dict``.
    """

    LABEL_COLS: list = []
    REG_COLS: list = []

    def __init__(self, df, base_dir: str | None, name: str, cols: dict):
        self._df, self._base_dir, self._name, self._cols = df, base_dir, name, cols

    def get_harmonized_df(self):
        return self._df

    def verify_images(self, drop_missing=True):
        from tqdm import tqdm

        def _exists(p):
            return os.path.isfile(os.path.join(self._base_dir, p) if self._base_dir else p)

        tqdm.pandas(desc=f"Verifying {self._name} images")
        mask = self._df["image_path"].progress_apply(_exists)
        missing = self._df[~mask].copy()
        if not missing.empty and drop_missing:
            self._df = self._df[mask].reset_index(drop=True)
        return missing


def _vqa_adapter(df, base_dir_exp: str | None, name: str) -> _TableAdapter:
    """Wrap a VQA table for the app: one sample per question, its Q&A in ``qa_text``.

    The app shows the same samples as the VQA dataset class, so a row shows
    only its own question and answer (D-11, Q-3).

    Args:
        df: Harmonized VQA table (``image_path``, ``question``, ``answer``).
        base_dir_exp: Folder that a relative ``image_path`` is relative to.
        name: Dataset name for the progress bar.
    """
    if base_dir_exp:
        # load_dataset puts base_dir in front of image_path. A table with an absolute
        # path (for example an old CheXinstruct table) is made relative first.
        df = df.assign(image_path=[
            os.path.relpath(p, base_dir_exp) if os.path.isabs(p) else p
            for p in df["image_path"].astype(str)
        ])
    qtypes = df["question_type"].astype(str) if "question_type" in df else ["question"] * len(df)
    structs = df["answer_struct"] if "answer_struct" in df else [None] * len(df)
    df = df.assign(qa_text=[
        f"Q ({t}): {q}\n   → {_answer_text(a, s)}"
        for t, q, a, s in zip(qtypes, df["question"].astype(str), df["answer"], structs, strict=True)
    ]).reset_index(drop=True)
    return _TableAdapter(df, base_dir_exp, name, {"qa_text": "qa_text"})


def _answer_text(answer, struct) -> str:
    """Return the answer to show: ``answer``, or the true options of ``answer_struct`` when it is empty.

    A CheXinstruct close-ended row keeps its answer only in ``answer_struct``, for example
    ``{"yes": True, "no": False}`` gives "yes" (V-A-45).
    """
    text = "" if answer is None or (isinstance(answer, float) and np.isnan(answer)) else str(answer)
    if not text.strip() and isinstance(struct, dict):
        text = ", ".join(str(k) for k, v in struct.items() if v is True)
    return text


def _build_vqa_rad_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.dataset import VQARadTrainDataset
    return _build_vqa_rad_cls(VQARadTrainDataset, base_dir, csv_path)


def _build_vqa_rad_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.dataset import VQARadTestDataset
    return _build_vqa_rad_cls(VQARadTestDataset, base_dir, csv_path)


def _build_vqa_rad_cls(cls, base_dir, csv_path):
    """Load VQA-RAD with the dataset class ``cls`` and wrap the table for the app."""
    base = os.path.expanduser(base_dir)
    try:
        df = cls(
            base_image_dir=base,
            json_path=os.path.expanduser(csv_path) if csv_path else None,
            cache_dir=None,
        ).get_harmonized_df()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"
    return _vqa_adapter(df, base, "VQA-RAD"), None


def _build_chexinstruct_vqa_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.dataset import ChexInstructVQATrainDataset
    return _build_chexinstruct_vqa_cls(ChexInstructVQATrainDataset, base_dir, csv_path, extra_field)


def _build_chexinstruct_vqa_val(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.dataset import ChexInstructVQAValDataset
    return _build_chexinstruct_vqa_cls(ChexInstructVQAValDataset, base_dir, csv_path, extra_field)


def _build_chexinstruct_vqa_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.dataset import ChexInstructVQATestDataset
    return _build_chexinstruct_vqa_cls(ChexInstructVQATestDataset, base_dir, csv_path, extra_field)


def _build_chexinstruct_vqa_cls(cls, base_dir, csv_path, extra_field):
    """Load CheXinstruct VQA with the dataset class ``cls`` and wrap the table for the app."""
    if not csv_path:
        return None, "CheXinstruct needs the compiled chexinstruct_vqa.json in the CSV path field."
    base = os.path.expanduser(base_dir)
    sources = [s.strip() for s in extra_field.split(",") if s.strip()] or None
    try:
        df = cls(
            base_image_dir=base,
            json_path=os.path.expanduser(csv_path),
            data_sources=sources,
            cache_dir=None,
        ).get_harmonized_df()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"
    return _vqa_adapter(df, base, "CheXinstruct"), None


# The QA zip is large, so the app loads only some studies by default.
_QBA_DEFAULT_STUDIES = 200


def _build_mimic_ext_cxr_qba_train(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.dataset import MIMICExtCXRQBATrainDataset
    return _build_mimic_ext_cxr_qba_cls(MIMICExtCXRQBATrainDataset, base_dir, csv_path, extra_field, extra_field2)


def _build_mimic_ext_cxr_qba_validate(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.dataset import MIMICExtCXRQBAValidateDataset
    return _build_mimic_ext_cxr_qba_cls(MIMICExtCXRQBAValidateDataset, base_dir, csv_path, extra_field, extra_field2)


def _build_mimic_ext_cxr_qba_test(base_dir, csv_path, extra_field, extra_field2, cache_dir, **flags):
    from radharmony.dataset import MIMICExtCXRQBATestDataset
    return _build_mimic_ext_cxr_qba_cls(MIMICExtCXRQBATestDataset, base_dir, csv_path, extra_field, extra_field2)


def _build_mimic_ext_cxr_qba_cls(cls, base_dir, csv_path, extra_field, extra_field2):
    """MIMIC-Ext-CXR-QBA: question-answer pairs for MIMIC-CXR studies, one official split (D-09).

    base_dir     → MIMIC-CXR files/ root (DICOM)
    csv_path     → qa.zip (required)
    extra_field  → metadata folder (required)
    extra_field2 → number of studies of the split to load (default _QBA_DEFAULT_STUDIES)
    """
    if not csv_path or not extra_field:
        return None, "MIMIC-Ext-CXR-QBA needs qa.zip (CSV path field) and the metadata folder."
    try:
        max_studies = int(extra_field2) if extra_field2 else _QBA_DEFAULT_STUDIES
    except ValueError:
        return None, f"Number of studies must be an integer, not {extra_field2!r}."
    base = os.path.expanduser(base_dir)
    try:
        df = cls(
            base_image_dir=base,
            metadata_dir=os.path.expanduser(extra_field),
            qa_zip_path=os.path.expanduser(csv_path),
            max_studies=max_studies,
            cache_dir=None,
        ).get_harmonized_df()
    except Exception as exc:
        return None, f"Harmonization error: {exc}"
    return _vqa_adapter(df, base, "MIMIC-Ext-CXR-QBA"), None


# The modality dropdown of the ROCOv2 entries.
_ROCOV2_MODALITIES = ("(all)", "ct", "mri", "xray", "ultrasound",
                      "angiography", "pet_spect", "mammography")

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
    "CheXpert-Plus (Train)": DatasetConfig(
        False,
        "Label JSON (optional)",
        _build_chexpert_plus_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/chexpertplus/DICOM/Uncompressed/",
        csv_placeholder="auto: df_chexpert_plus_240401.csv",
        extra_placeholder="auto: impression_fixed.json",
    ),
    "CheXpert-Plus (Valid)": DatasetConfig(
        False,
        "Label JSON (optional)",
        _build_chexpert_plus_valid,
        modality="CXR",
        base_dir_placeholder="e.g. /data/chexpertplus/DICOM/Uncompressed/",
        csv_placeholder="auto: df_chexpert_plus_240401.csv",
        extra_placeholder="auto: impression_fixed.json",
    ),
    "CheXlocalize (Val)": DatasetConfig(
        False,
        "Segmentation JSON (optional)",
        _build_chexlocalize_val,
        modality="CXR",
        extra_field2_label="Mask output dir (required for output_mask)",
        base_dir_placeholder="e.g. /data/chexlocalize/CheXpert/val/",
        csv_placeholder="auto: val_labels.csv",
        extra_placeholder="auto: gt_segmentations_val.json",
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
    "── CheXphoto ──": _header("CXR"),
    "CheXphoto (Train)": DatasetConfig(
        False,
        "",
        _build_chexphoto_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/CheXphoto/  (contains natural/ and synthetic/)",
        csv_placeholder="auto: train.csv",
        extra_dropdown_label="Subset",
        extra_dropdown_choices=("(all)", "natural", "synthetic"),
        extra_dropdown_kwarg="chexphoto_subset",
    ),
    "CheXphoto (Valid)": DatasetConfig(
        False,
        "",
        _build_chexphoto_valid,
        modality="CXR",
        base_dir_placeholder="e.g. /data/CheXphoto/  (contains natural/ and synthetic/)",
        csv_placeholder="auto: valid.csv",
        extra_dropdown_label="Subset",
        extra_dropdown_choices=("(all)", "natural", "synthetic"),
        extra_dropdown_kwarg="chexphoto_subset",
    ),
    "── ChestX-ray14 ──": _header("CXR"),
    "ChestX-ray14 (Train)": DatasetConfig(
        False,
        "",
        _build_chestxray14_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/NIH_CXR/CXR14/",
        csv_placeholder="auto: Data_Entry_2017.csv",
    ),
    "ChestX-ray14 (Test)": DatasetConfig(
        False,
        "",
        _build_chestxray14_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/NIH_CXR/CXR14/",
        csv_placeholder="auto: Data_Entry_2017.csv",
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
    "── ChestX-Det10 ──": _header("CXR"),
    "ChestX-Det10 (Train)": DatasetConfig(
        False,
        "",
        _build_chestxdet10_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/ChestX-Det10/  (dir containing train-old/ and test_data/)",
        csv_placeholder="auto: train.json",
        bbox_default=True,
    ),
    "ChestX-Det10 (Test)": DatasetConfig(
        False,
        "",
        _build_chestxdet10_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/ChestX-Det10/  (dir containing train-old/ and test_data/)",
        csv_placeholder="auto: test.json",
        bbox_default=True,
    ),
    "── NODE21 ──": _header("CXR"),
    "NODE21 (processed)": DatasetConfig(
        False,
        "",
        _build_node21_processed,
        modality="CXR",
        base_dir_placeholder="e.g. /data/NODE21/  (dir containing cxr_images/)",
        csv_placeholder="auto: cxr_images/proccessed_data/metadata.csv",
        bbox_default=True,
    ),
    "NODE21 (original)": DatasetConfig(
        False,
        "",
        _build_node21_original,
        modality="CXR",
        base_dir_placeholder="e.g. /data/NODE21/  (dir containing cxr_images/)",
        csv_placeholder="auto: cxr_images/original_data/metadata.csv",
        bbox_default=True,
    ),
    "── Emory CXR v2 ──": _header("CXR"),
    "Emory CXR v2": DatasetConfig(
        False,
        "Finding-label CSV (optional)",
        _build_emory_cxr,
        modality="CXR",
        extra_field2_label="Report CSV (optional)",
        base_dir_placeholder="e.g. /data/EmoryCXRv2/DEID_PNG/",
        csv_placeholder="EmoryCXR_v2_Metadata_<date>.csv (pin one; not interchangeable)",
        extra_placeholder="e.g. .../TABLES/EmoryCXR_v2_FindingLabel_10162024.csv",
        extra_placeholder2="e.g. .../TABLES/EmoryCXR_v2_Report_05102026.csv",
    ),
    "── Emory CHORUS ──": _header("CXR"),
    "Emory CHORUS (X-ray, chest only)": DatasetConfig(
        False,
        "",
        _build_emory_chorus,
        modality="CXR",
        csv_label="Manifest CSV (optional scan cache, written on the first build)",
        base_dir_placeholder="e.g. /data/Emory_CHORUS/images_dicom_xray/",
        csv_placeholder="e.g. /data/Emory_CHORUS/chorus_xray_manifest.csv",
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
    "── PadChest-GR ──": _header("CXR"),
    "PadChest-GR (Train)": DatasetConfig(
        False,
        "PadChest labels CSV (optional; fills view_position)",
        _build_padchest_gr_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/PadChest-GR/  (dir containing PadChest_GR/ and grounded_reports_*.json)",
        csv_placeholder="auto: master_table.csv",
        extra_placeholder="e.g. .../PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
        bbox_default=True,
    ),
    "PadChest-GR (Validation)": DatasetConfig(
        False,
        "PadChest labels CSV (optional; fills view_position)",
        _build_padchest_gr_validation,
        modality="CXR",
        base_dir_placeholder="e.g. /data/PadChest-GR/  (dir containing PadChest_GR/ and grounded_reports_*.json)",
        csv_placeholder="auto: master_table.csv",
        extra_placeholder="e.g. .../PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
        bbox_default=True,
    ),
    "PadChest-GR (Test)": DatasetConfig(
        False,
        "PadChest labels CSV (optional; fills view_position)",
        _build_padchest_gr_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/PadChest-GR/  (dir containing PadChest_GR/ and grounded_reports_*.json)",
        csv_placeholder="auto: master_table.csv",
        extra_placeholder="e.g. .../PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz",
        bbox_default=True,
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
    "MIMIC-CXR (Train)": DatasetConfig(
        False,
        "Label CSV (needed: it sits next to the split file)",
        _build_mimic_train,
        modality="CXR",
        extra_field2_label="Report CSV (optional)",
        base_dir_placeholder="e.g. /data/mimic-cxr/2.1.0/files/",
        csv_placeholder="auto: cxr-record-list.csv.gz",
        extra_placeholder="from MIMIC-CXR-JPG: mimic-cxr-2.0.0-chexpert.csv",
        extra_placeholder2="auto: cxr-study-list.csv.gz",
    ),
    "MIMIC-CXR (Validate)": DatasetConfig(
        False,
        "Label CSV (needed: it sits next to the split file)",
        _build_mimic_validate,
        modality="CXR",
        extra_field2_label="Report CSV (optional)",
        base_dir_placeholder="e.g. /data/mimic-cxr/2.1.0/files/",
        csv_placeholder="auto: cxr-record-list.csv.gz",
        extra_placeholder="from MIMIC-CXR-JPG: mimic-cxr-2.0.0-chexpert.csv",
        extra_placeholder2="auto: cxr-study-list.csv.gz",
    ),
    "MIMIC-CXR (Test)": DatasetConfig(
        False,
        "Label CSV (needed: it sits next to the split file)",
        _build_mimic_test,
        modality="CXR",
        extra_field2_label="Report CSV (optional)",
        base_dir_placeholder="e.g. /data/mimic-cxr/2.1.0/files/",
        csv_placeholder="auto: cxr-record-list.csv.gz",
        extra_placeholder="from MIMIC-CXR-JPG: mimic-cxr-2.0.0-chexpert.csv",
        extra_placeholder2="auto: cxr-study-list.csv.gz",
    ),
    "MIMIC-CXR-JPG (Train)": DatasetConfig(
        False,
        "Label CSV (optional)",
        _build_mimic_jpg_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
        csv_placeholder="auto: mimic-cxr-2.0.0-metadata.csv.gz",
        extra_placeholder="auto: mimic-cxr-2.0.0-chexpert.csv",
    ),
    "MIMIC-CXR-JPG (Validate)": DatasetConfig(
        False,
        "Label CSV (optional)",
        _build_mimic_jpg_validate,
        modality="CXR",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
        csv_placeholder="auto: mimic-cxr-2.0.0-metadata.csv.gz",
        extra_placeholder="auto: mimic-cxr-2.0.0-chexpert.csv",
    ),
    "MIMIC-CXR-JPG (Test, labeler labels)": DatasetConfig(
        False,
        "Label CSV (optional)",
        _build_mimic_jpg_test_chexpert,
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
    "RANZCR CLiP (Train)": DatasetConfig(
        False,
        "Annotations CSV (optional)",
        _build_ranzcr_clip_train,
        modality="CXR",
        extra_field2_label="Mask output dir (required for output_mask)",
        base_dir_placeholder="e.g. ~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/",
        csv_placeholder="auto: train.csv",
        extra_placeholder="auto: train_annotations.csv",
    ),
    "RANZCR CLiP (Test)": DatasetConfig(
        False,
        "",
        _build_ranzcr_clip_test,
        modality="CXR",
        base_dir_placeholder="e.g. ~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/",
        csv_placeholder="not used: lists test/*.jpg",
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
    "TAIX-Ray (512px, Train)": DatasetConfig(
        False,
        "",
        _build_taix_ray_512_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/TAIX-Ray/data_512/images/",
        csv_placeholder="auto: annotation.csv",
        extra_dropdown_label="Label mode",
        extra_dropdown_choices=("binary", "ordinal"),
        extra_dropdown_kwarg="label_mode",
    ),
    "TAIX-Ray (512px, Val)": DatasetConfig(
        False,
        "",
        _build_taix_ray_512_val,
        modality="CXR",
        base_dir_placeholder="e.g. /data/TAIX-Ray/data_512/images/",
        csv_placeholder="auto: annotation.csv",
        extra_dropdown_label="Label mode",
        extra_dropdown_choices=("binary", "ordinal"),
        extra_dropdown_kwarg="label_mode",
    ),
    "TAIX-Ray (512px, Test)": DatasetConfig(
        False,
        "",
        _build_taix_ray_512_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/TAIX-Ray/data_512/images/",
        csv_placeholder="auto: annotation.csv",
        extra_dropdown_label="Label mode",
        extra_dropdown_choices=("binary", "ordinal"),
        extra_dropdown_kwarg="label_mode",
    ),
    "TAIX-Ray (original, Train)": DatasetConfig(
        False,
        "",
        _build_taix_ray_original_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/TAIX-Ray/data_original/images/",
        csv_placeholder="auto: annotation.csv",
        extra_dropdown_label="Label mode",
        extra_dropdown_choices=("binary", "ordinal"),
        extra_dropdown_kwarg="label_mode",
    ),
    "TAIX-Ray (original, Val)": DatasetConfig(
        False,
        "",
        _build_taix_ray_original_val,
        modality="CXR",
        base_dir_placeholder="e.g. /data/TAIX-Ray/data_original/images/",
        csv_placeholder="auto: annotation.csv",
        extra_dropdown_label="Label mode",
        extra_dropdown_choices=("binary", "ordinal"),
        extra_dropdown_kwarg="label_mode",
    ),
    "TAIX-Ray (original, Test)": DatasetConfig(
        False,
        "",
        _build_taix_ray_original_test,
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
    "VinDr-PCXR (Train)": DatasetConfig(
        False,
        "",
        _build_vindr_pcxr_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/VINDR-PCXR/",
        csv_placeholder="(not required, parsed from image_labels CSV files)",
    ),
    "VinDr-PCXR (Test)": DatasetConfig(
        False,
        "",
        _build_vindr_pcxr_test,
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
    "── QaTa-COV19 ──": _header("CXR"),
    "QaTa-COV19 v2 (Train)": DatasetConfig(
        False,
        "Ground-truths dir (optional — defaults to sibling)",
        _build_qata_cov19_train,
        modality="CXR",
        base_dir_placeholder="e.g. /data/QaTa-COV19/QaTa-COV19-v2/Train Set/Images/",
        csv_placeholder="(not required — no CSV in the release)",
        extra_placeholder="e.g. /data/QaTa-COV19/QaTa-COV19-v2/Train Set/Ground-truths/",
    ),
    "QaTa-COV19 v2 (Test)": DatasetConfig(
        False,
        "Ground-truths dir (optional — defaults to sibling)",
        _build_qata_cov19_test,
        modality="CXR",
        base_dir_placeholder="e.g. /data/QaTa-COV19/QaTa-COV19-v2/Test Set/Images/",
        csv_placeholder="(not required — no CSV in the release)",
        extra_placeholder="e.g. /data/QaTa-COV19/QaTa-COV19-v2/Test Set/Ground-truths/",
    ),
    "── MedSeg COVID CT ──": _header("CT (2D)"),
    "MedSeg COVID-19 CT (val_im, no masks)": DatasetConfig(
        False,
        "Slice output dir (required)",
        _build_medseg_covid_ct_val,
        modality="CT (2D)",
        base_dir_placeholder="e.g. /data/MedSeg-COVID-CT/figshare_13521488",
        csv_placeholder="(not required — read from val_im.nii.gz)",
        extra_placeholder="e.g. /data/MedSeg-COVID-CT/_slices",
    ),
    "MedSeg COVID-19 CT": DatasetConfig(
        False,
        "Slice output dir (required)",
        _build_medseg_covid_ct,
        modality="CT (2D)",
        extra_field2_label="Subset (dataset1 | dataset2)",
        base_dir_placeholder="e.g. /data/MedSeg-COVID-CT/figshare_13521488",
        csv_placeholder="(not required — read from the NIfTI volumes)",
        extra_placeholder="e.g. /data/MedSeg-COVID-CT/_slices",
        extra_placeholder2="dataset1",
    ),
    "── MS-CXR ──": _header("CXR"),
    "MS-CXR (phrase grounding, Train)": DatasetConfig(
        False,
        "",
        _build_ms_cxr_train,
        modality="CXR",
        csv_label="CSV path (required — MS-CXR local alignment CSV)",
        base_dir_placeholder="MIMIC-CXR files/ root (JPG or DICOM), e.g. /data/MIMIC-CXR-V2-AWS/files/",
        csv_placeholder="e.g. /data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv",
    ),
    "MS-CXR (phrase grounding, Val)": DatasetConfig(
        False,
        "",
        _build_ms_cxr_val,
        modality="CXR",
        csv_label="CSV path (required — MS-CXR local alignment CSV)",
        base_dir_placeholder="MIMIC-CXR files/ root (JPG or DICOM), e.g. /data/MIMIC-CXR-V2-AWS/files/",
        csv_placeholder="e.g. /data/ms-cxr/MS_CXR_Local_Alignment_v1.1.0.csv",
    ),
    "MS-CXR (phrase grounding, Test)": DatasetConfig(
        False,
        "",
        _build_ms_cxr_test,
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
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
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
        bbox_default=True,
    ),
    "Chest ImaGenome (Silver, Train)": DatasetConfig(
        False,
        "Annotation dir (Chest ImaGenome root)",
        _build_chest_imagenome_silver_train,
        modality="CXR",
        base_dir_placeholder="MIMIC-CXR DICOM files/ root, e.g. /data/MIMIC-CXR-V2-AWS/files/",
        csv_label="(not used - annotations come from the annotation dir)",
        extra_placeholder="e.g. /data/CHEST-IMAGENOME/",
        bbox_default=True,
    ),
    "Chest ImaGenome (Silver, Valid)": DatasetConfig(
        False,
        "Annotation dir (Chest ImaGenome root)",
        _build_chest_imagenome_silver_valid,
        modality="CXR",
        base_dir_placeholder="MIMIC-CXR DICOM files/ root, e.g. /data/MIMIC-CXR-V2-AWS/files/",
        csv_label="(not used - annotations come from the annotation dir)",
        extra_placeholder="e.g. /data/CHEST-IMAGENOME/",
        bbox_default=True,
    ),
    "Chest ImaGenome (Silver, Test)": DatasetConfig(
        False,
        "Annotation dir (Chest ImaGenome root)",
        _build_chest_imagenome_silver_test,
        modality="CXR",
        base_dir_placeholder="MIMIC-CXR DICOM files/ root, e.g. /data/MIMIC-CXR-V2-AWS/files/",
        csv_label="(not used - annotations come from the annotation dir)",
        extra_placeholder="e.g. /data/CHEST-IMAGENOME/",
        bbox_default=True,
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
    "CT-RATE (Train)": DatasetConfig(
        True,
        "Metadata CSV (optional)",
        _build_ctrate_train,
        modality="CT",
        base_dir_placeholder="e.g. /data/CT-RATE/dataset/train_fixed/",
        csv_placeholder="auto: train_predicted_labels.csv",
        extra_placeholder="auto: train_metadata.csv",
    ),
    "CT-RATE (Valid)": DatasetConfig(
        True,
        "Metadata CSV (optional)",
        _build_ctrate_valid,
        modality="CT",
        base_dir_placeholder="e.g. /data/CT-RATE/dataset/valid_fixed/",
        csv_placeholder="auto: valid_predicted_labels.csv",
        extra_placeholder="auto: validation_metadata.csv",
    ),
    "── RAD-ChestCT ──": _header("CT"),
    "RAD-ChestCT (Train)": DatasetConfig(
        True,
        "Label CSV (optional)",
        _build_radchestct_train,
        modality="CT",
        extra_field2_label="BBox CSV (optional)",
        base_dir_placeholder="e.g. /data/RAD-ChestCT/images/",
        csv_placeholder="auto: CT_Scan_Metadata_Complete_35747.csv",
        extra_placeholder="auto: imgtrain_Abnormality_and_Location_Labels.csv",
    ),
    "RAD-ChestCT (Valid)": DatasetConfig(
        True,
        "Label CSV (optional)",
        _build_radchestct_valid,
        modality="CT",
        extra_field2_label="BBox CSV (optional)",
        base_dir_placeholder="e.g. /data/RAD-ChestCT/images/",
        csv_placeholder="auto: CT_Scan_Metadata_Complete_35747.csv",
        extra_placeholder="auto: imgvalid_Abnormality_and_Location_Labels.csv",
    ),
    "RAD-ChestCT (Test)": DatasetConfig(
        True,
        "Label CSV (optional)",
        _build_radchestct_test,
        modality="CT",
        extra_field2_label="BBox CSV (optional)",
        base_dir_placeholder="e.g. /data/RAD-ChestCT/images/",
        csv_placeholder="auto: CT_Scan_Metadata_Complete_35747.csv",
        extra_placeholder="auto: imgtest_Abnormality_and_Location_Labels.csv",
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
        "",  # no extra field: the CSV box holds the series metadata (V-A-36)
        _build_rsna_abdominal_trauma_2023_test,
        modality="CT",
        csv_label="Series metadata CSV (optional)",
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
        "",  # no extra field: the CSV box holds the series descriptions (V-A-36)
        _build_rsna_2024_lumbar_spine_test,
        modality="MRI",
        csv_label="Series descriptions CSV (optional)",
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
    "RSNA Bone Age (Test, no labels)": DatasetConfig(
        False,
        "",
        _build_rsna_bone_age_test,
        modality="Radiograph",
        base_dir_placeholder="e.g. ~/datasets/boneage-test-dataset/",
        csv_placeholder="not used: lists the PNGs",
    ),
    # ── VQA datasets ─────────────────────────────────────────────────────────
    "── ROCO ──": _header("VQA"),
    "ROCO (radiology captioning, Train)": DatasetConfig(
        False,
        "",
        _build_roco_train,
        modality="VQA",
        base_dir_placeholder="e.g. /data/roco-dataset/",
        csv_placeholder="(not required)",
    ),
    "ROCO (radiology captioning, Validation)": DatasetConfig(
        False,
        "",
        _build_roco_validation,
        modality="VQA",
        base_dir_placeholder="e.g. /data/roco-dataset/",
        csv_placeholder="(not required)",
    ),
    "ROCO (radiology captioning, Test)": DatasetConfig(
        False,
        "",
        _build_roco_test,
        modality="VQA",
        base_dir_placeholder="e.g. /data/roco-dataset/",
        csv_placeholder="(not required)",
    ),
    "── ROCOv2 ──": _header("VQA"),
    "ROCOv2 (radiology, curated UMLS modality, Train)": DatasetConfig(
        False,
        "",
        _build_rocov2_train,
        modality="VQA",
        base_dir_placeholder="e.g. /data/ROCOv2/",
        csv_placeholder="(not required)",
        extra_dropdown_label="Modality",
        extra_dropdown_choices=_ROCOV2_MODALITIES,
        extra_dropdown_kwarg="rocov2_modality",
    ),
    "ROCOv2 (radiology, curated UMLS modality, Validation)": DatasetConfig(
        False,
        "",
        _build_rocov2_validation,
        modality="VQA",
        base_dir_placeholder="e.g. /data/ROCOv2/",
        csv_placeholder="(not required)",
        extra_dropdown_label="Modality",
        extra_dropdown_choices=_ROCOV2_MODALITIES,
        extra_dropdown_kwarg="rocov2_modality",
    ),
    "ROCOv2 (radiology, curated UMLS modality, Test)": DatasetConfig(
        False,
        "",
        _build_rocov2_test,
        modality="VQA",
        base_dir_placeholder="e.g. /data/ROCOv2/",
        csv_placeholder="(not required)",
        extra_dropdown_label="Modality",
        extra_dropdown_choices=_ROCOV2_MODALITIES,
        extra_dropdown_kwarg="rocov2_modality",
    ),
    "── MR-RATE ──": _header("MRI"),
    "MR-RATE (brain/spine MRI + reports)": DatasetConfig(
        True,                                   # 3-D NIfTI volumes
        "",
        _build_mrrate,
        modality="MRI",
        base_dir_placeholder="e.g. /data/MR-RATE/",
        csv_placeholder="(not required)",
        extra_placeholder="splits (default: test)",
        extra_dropdown_label="Sequence",
        extra_dropdown_choices=("(all)", "T1w", "T2w", "FLAIR", "SWI", "MRA"),
        extra_dropdown_kwarg="mrrate_modality",
    ),
    "── MedPMC ──": _header("VQA"),
    "MedPMC (PMC, OUR derived tags)": DatasetConfig(
        False,
        "",
        _build_medpmc,
        modality="VQA",
        base_dir_placeholder="e.g. /data/MedPMC/",
        csv_placeholder="(not required)",
        extra_placeholder=("CT MR XR US XA PT/NM MG   (blank = all; "
                           "lowercase ct mri xray ... also accepted)"),
        extra_dropdown_label="Tag precision",
        extra_dropdown_choices=("high-precision radiology",
                                "human-verified only",
                                "radiology (both taggers agree)",
                                "radiology (caption tagger only)"),
        extra_dropdown_kwarg="medpmc_preset",
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
    "── VQA-RAD ──": _header("VQA"),
    "VQA-RAD (Train)": DatasetConfig(
        False,
        "",
        _build_vqa_rad_train,
        modality="VQA",
        csv_label="JSON path (optional)",
        base_dir_placeholder="e.g. /data/VQA-RAD/VQA_RAD Image Folder/",
        csv_placeholder="auto: VQA_RAD Dataset Public.json near the image folder",
    ),
    "VQA-RAD (Test)": DatasetConfig(
        False,
        "",
        _build_vqa_rad_test,
        modality="VQA",
        csv_label="JSON path (optional)",
        base_dir_placeholder="e.g. /data/VQA-RAD/VQA_RAD Image Folder/",
        csv_placeholder="auto: VQA_RAD Dataset Public.json near the image folder",
    ),
    "── CheXinstruct ──": _header("VQA"),
    "CheXinstruct (VQA, Train)": DatasetConfig(
        False,
        "Data sources (optional)",
        _build_chexinstruct_vqa_train,
        modality="VQA",
        csv_label="JSON path (required, compiled chexinstruct_vqa.json)",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
        csv_placeholder="e.g. /data/CheXinstruct/compiled/chexinstruct_vqa.json",
        extra_placeholder="comma-separated, e.g. MIMIC-CXR-VQA (blank = all)",
    ),
    "CheXinstruct (VQA, Val)": DatasetConfig(
        False,
        "Data sources (optional)",
        _build_chexinstruct_vqa_val,
        modality="VQA",
        csv_label="JSON path (required, compiled chexinstruct_vqa.json)",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
        csv_placeholder="e.g. /data/CheXinstruct/compiled/chexinstruct_vqa.json",
        extra_placeholder="comma-separated, e.g. MIMIC-CXR-VQA (blank = all)",
    ),
    "CheXinstruct (VQA, Test)": DatasetConfig(
        False,
        "Data sources (optional)",
        _build_chexinstruct_vqa_test,
        modality="VQA",
        csv_label="JSON path (required, compiled chexinstruct_vqa.json)",
        base_dir_placeholder="e.g. /data/mimic-cxr-jpg/2.0.0/files/",
        csv_placeholder="e.g. /data/CheXinstruct/compiled/chexinstruct_vqa.json",
        extra_placeholder="comma-separated, e.g. MIMIC-CXR-VQA (blank = all)",
    ),
    "── MIMIC-Ext-CXR-QBA ──": _header("VQA"),
    "MIMIC-Ext-CXR-QBA (Train)": DatasetConfig(
        False,
        "Metadata folder (required)",
        _build_mimic_ext_cxr_qba_train,
        modality="VQA",
        extra_field2_label="Number of studies",
        csv_label="QA zip path (required, qa.zip)",
        base_dir_placeholder="e.g. /data/MIMIC-CXR/files/ (DICOM)",
        csv_placeholder="e.g. /data/MIMIC_EXT_CXR_QBA/qa.zip",
        extra_placeholder="e.g. /data/MIMIC_EXT_CXR_QBA/exports/A_frontal/metadata/q1M/",
        extra_placeholder2=f"default {_QBA_DEFAULT_STUDIES}",
    ),
    "MIMIC-Ext-CXR-QBA (Validate)": DatasetConfig(
        False,
        "Metadata folder (required)",
        _build_mimic_ext_cxr_qba_validate,
        modality="VQA",
        extra_field2_label="Number of studies",
        csv_label="QA zip path (required, qa.zip)",
        base_dir_placeholder="e.g. /data/MIMIC-CXR/files/ (DICOM)",
        csv_placeholder="e.g. /data/MIMIC_EXT_CXR_QBA/qa.zip",
        extra_placeholder="e.g. /data/MIMIC_EXT_CXR_QBA/exports/A_frontal/metadata/q1M/",
        extra_placeholder2=f"default {_QBA_DEFAULT_STUDIES}",
    ),
    "MIMIC-Ext-CXR-QBA (Test)": DatasetConfig(
        False,
        "Metadata folder (required)",
        _build_mimic_ext_cxr_qba_test,
        modality="VQA",
        extra_field2_label="Number of studies",
        csv_label="QA zip path (required, qa.zip)",
        base_dir_placeholder="e.g. /data/MIMIC-CXR/files/ (DICOM)",
        csv_placeholder="e.g. /data/MIMIC_EXT_CXR_QBA/qa.zip",
        extra_placeholder="e.g. /data/MIMIC_EXT_CXR_QBA/exports/A_frontal/metadata/q1M/",
        extra_placeholder2=f"default {_QBA_DEFAULT_STUDIES}",
    ),
}

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

    Each ``── ... ──`` header starts a new family in its modality. The next
    non-header keys belong to that family.  Every key needs a header before it.
    A key without a header for its modality is a registry mistake and raises.
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


# Optional dataset plugins. Each package named in the RADHARMONY_APP_PLUGINS
# env var (comma-separated) may expose a callable
# ``register_app_datasets(registry, DatasetConfig, _header, app_dtype)`` that
# appends entries to DATASET_REGISTRY. Lets third-party / private dataset
# packages extend the app without editing it.
for _plugin in os.environ.get("RADHARMONY_APP_PLUGINS", "").split(","):
    _plugin = _plugin.strip()
    if not _plugin:
        continue
    _before = dict(DATASET_REGISTRY)
    try:
        _mod = importlib.import_module(_plugin)
        _mod.register_app_datasets(DATASET_REGISTRY, DatasetConfig, _header, _APP_DTYPE)
        # An entry with no family header in front of it stops the app (V-A-42).
        _split_into_families_per_modality(DATASET_REGISTRY)
    except Exception as _e:  # noqa: BLE001 — a bad plugin shouldn't kill the app
        DATASET_REGISTRY.clear()
        DATASET_REGISTRY.update(_before)
        print(f"[radharmony] app plugin {_plugin!r} not loaded: {_e}")


_MODALITY_GROUPS = _group_by_modality(DATASET_REGISTRY)
_MODALITY_FAMILIES = _split_into_families_per_modality(DATASET_REGISTRY)


# ---------------------------------------------------------------------------
# Image rendering helpers
# ---------------------------------------------------------------------------


def _to_rgb(arr01: np.ndarray) -> np.ndarray:
    """(H,W) float [0,1] → (H,W,3) uint8."""
    g = (arr01 * 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


def _bbox_class_vocab(ds_state) -> list[str]:
    """Sorted unique bbox class names across a loaded dataset.

    Reads the pre-transform ``data_dicts`` so the vocabulary is known before
    any sample is drawn.  Returns [] when the dataset carries no box labels.
    """
    if not ds_state:
        return []
    seen = set()
    for d in ds_state.get("data_dicts", []) or []:
        for lab in _parse_str_list(d.get("bbox_labels")) or []:
            if lab:
                seen.add(str(lab))
    return sorted(seen)


def _filter_bbox_by_class(sample_dict: dict, keep) -> dict:
    """Drop boxes whose label is not in ``keep``.

    ``bbox``, ``bbox_labels`` and ``bbox_findings`` are parallel arrays.
    Filter them with one shared index set. Dropping from one alone would
    misalign the other labels.  An empty or None ``keep`` means "show all".
    This is also the result when the dataset has no labels to filter on.
    """
    import json as _json

    keep = set(keep or [])
    if not keep:
        return sample_dict
    labels = _parse_str_list(sample_dict.get("bbox_labels")) or []
    boxes = _parse_str_list(sample_dict.get("bbox")) or []
    if not labels or not boxes:
        return sample_dict
    idx = [i for i, l in enumerate(labels) if str(l) in keep]
    if len(idx) == len(labels):
        return sample_dict
    out = dict(sample_dict)
    out["bbox"] = _json.dumps([boxes[i] for i in idx if i < len(boxes)])
    out["bbox_labels"] = _json.dumps([labels[i] for i in idx])
    finds = _parse_str_list(sample_dict.get("bbox_findings"))
    if finds:
        out["bbox_findings"] = _json.dumps(
            [finds[i] for i in idx if i < len(finds)]
        )
    return out


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
            instead of a rectangle outline.  Useful for dense, overlapping
            bbox sets (for example, spine fracture or lumbar coordinates).
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
    # The dot radius scales with image height. Small dots keep dense clusters
    # readable. The dots must still show on big panels.
    dot_radius = max(1, min(3, H // 150))
    for idx, b in enumerate(boxes):
        if any(not (0.0 <= v <= 1.0) for v in b):
            continue
        # Skip an all-zero box. Without this skip, the inflate-to-min-size logic
        # below renders a tiny stub box at the origin.
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
            # Make tiny rectangles bigger. Thin axes then show on coronal
            # and sagittal views (lumbar D is about 25 slices, so the
            # bbox z-extent is about 1 voxel).
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
        "Chest ImaGenome (Silver, Train)",
        "Chest ImaGenome (Silver, Valid)",
        "Chest ImaGenome (Silver, Test)",
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
    anatomy.  Returns ``labels`` unchanged when there are no findings (for
    example, every other bbox dataset). So it does nothing outside Chest ImaGenome.
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
    position ``bbox_idx`` (modulo n_boxes). The "Next bbox →" button uses this
    to cycle through annotations without a new pipeline run.
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
        # the FIRST bbox's center so the shown slices contain it.
        # Other bboxes are dropped from the overlay. This keeps the view on
        # one annotation, not every bbox on every panel.
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
    """Return (p_min, p_max): the param dicts at each extreme.

    The two dicts give a fixed side-by-side comparison."""
    lo, hi = dict(p), dict(p)
    if aug_name == "transpose":
        return lo, hi  # No random range, so no min or max.
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
        # Tensors are (C, H, W) in 2-D or (C, D, H, W) in 3-D.
        # Check now, to fail early with a clear message and not deep in MONAI.
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
            # 3-D Gaussian smoothing uses a kernel of size ~4*sigma+1 per axis.
            # Work and memory grow with sigma^3.
            # On a 112^3 volume, sigma > 10 can exceed the memory budget.
            # The OS then kills the process with no Python traceback. Fail early.
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
            # 3-D Rand3DElastic makes a full-volume displacement field,
            # smoothed by a Gaussian of `sigma`.  On a 112^3 volume, sigma > 50
            # or magnitude > 50 can push peak memory past 20 GB.
            # The OS then kills the process with no Python traceback.
            # Fail early with a clear message.
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
    single file).  Returns ``(x, y, z)`` as ITK gives it to the 3-D load, or
    ``None`` if unknown.  For a series dir, ITK reads only the file headers
    (no pixels).  For a single file, for example NIfTI, it uses MONAI's
    ITKReader.  Used to populate the "Native: ..." readout under the spacing
    sliders."""
    if not image_path:
        return None
    # ``image_path`` can start with ``~`` if the user typed an unexpanded base_dir.
    # os.path.exists does not expand ``~``.
    # Before, this looked like "spacing not available" on good datasets.
    image_path = os.path.expanduser(image_path)
    if not os.path.exists(image_path):
        return None

    if os.path.isdir(image_path):
        try:
            import itk

            # The same series and file order as MONAI's ITKReader in the 3-D load.
            # ITK takes z from the slice positions, not from SliceThickness, which
            # can differ, for example 0.625 for a 0.312 step (V-A-31).
            names = itk.GDCMSeriesFileNames.New()
            names.SetUseSeriesDetails(True)
            names.AddSeriesRestriction("0008|0021")
            names.SetDirectory(image_path)
            uids = names.GetSeriesUIDs()
            if not uids:
                return None
            files = names.GetFileNames(uids[0])
            reader = (itk.ImageFileReader.New(FileName=files[0]) if len(files) == 1
                      else itk.ImageSeriesReader.New(FileNames=files))
            reader.UpdateOutputInformation()
            sx, sy, sz = (float(v) for v in reader.GetOutput().GetSpacing())
            return (sx, sy, sz)
        except Exception:
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


_SPACING_MIN, _SPACING_MAX = 0.25, 5.0  # range of the 3-D spacing sliders (mm)


def _match_native_values(spacing) -> tuple[float, float, float]:
    """Native spacing (mm) rounded to 2 digits and clipped to the slider range."""
    return tuple(min(_SPACING_MAX, max(_SPACING_MIN, round(v, 2))) for v in spacing)


def _format_native_spacing(spacing: tuple[float, float, float] | None) -> str:
    """User-facing label for the "Native: ..." caption above the sliders."""
    if spacing is None:
        return "Native spacing: _not available for this dataset_"
    x, y, z = spacing
    return f"Native spacing: **{x:.2f} × {y:.2f} × {z:.2f} mm**"


#: The app entries of color figures. Their dataset classes set IMG_CHANNELS = 3 (D-42).
_COLOR_FIGURE_ENTRIES = ("ROCO ", "ROCOv2 ", "MedPMC ", "VQA-RAD ")


def _channels(dataset_name: str) -> int:
    """The channel count of ``img`` for an app entry (D-42)."""
    return 3 if dataset_name.startswith(_COLOR_FIGURE_ENTRIES) else 1


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


def _dropped_text(dropped) -> str:
    """Say how many rows ``verify_images()`` dropped, and why.

    Some datasets give the reason in a ``verify_reason`` column (for example
    ChestX-ray14: missing, unreadable). Without it, the reason
    is a missing file.
    """
    n = len(dropped) if dropped is not None else 0
    if not n:
        return ""
    if "verify_reason" in getattr(dropped, "columns", ()):
        counts = dropped["verify_reason"].value_counts()
        why = ", ".join(f"{reason} {count}" for reason, count in counts.items())
    else:
        why = "missing image files"
    return f"  ({n} rows dropped: {why})"


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
    progress=gr.Progress(track_tqdm=True),  # noqa: B008
):
    """Load any registered dataset. Dispatches via DATASET_REGISTRY."""
    # All four early-return paths must return 3 values to match the binding
    # in 3-D tabs: outputs=[state, load_status, native_spacing_md].  The
    # markdown placeholder is computed once here and reused.
    _no_native = _format_native_spacing(None)

    base_dir = (base_dir or "").strip()
    if not base_dir:
        return None, "Please provide a base image directory.", _no_native

    # Pre-flight check for typos, such as a missing "~/" or "/mnt/" prefix.
    # Later "<file> is required" messages cannot tell "directory missing"
    # from "file missing inside a good directory".
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

    # Expand "~" once here, so every build function gets a real path (V-A-33).
    def _path(v):
        return os.path.expanduser((v or "").strip())

    try:
        progress(0, desc="Building dataset…")
        ds_obj, err = config.build(
            os.path.expanduser(base_dir),
            _path(csv_path),
            _path(extra_field),
            _path(extra_field2),
            _path(cache_dir) or None,
            **build_flags,
        )
        if err:
            return None, err, _no_native

        progress(0.3, desc="Verifying image files…")
        missing = ds_obj.verify_images()

        progress(0.5, desc="Harmonizing metadata…")
        df = ds_obj.get_harmonized_df()
        progress(0.8, desc="Preparing data dicts…")
        # Join image_path to the folder the class reads from (D-04). For MedSeg it is
        # the slice folder, not the typed folder (V-A-30). An app adapter has none.
        data_dicts = get_data_dict(
            df, getattr(ds_obj, "base_image_dir", None) or base_dir, "image_path",
            ds_obj._cols, num_cores=1,
        )
        if not data_dicts:
            return None, "No sample to show: 0 samples." + _dropped_text(missing), _no_native

        available_keys = {"img"}
        for k in ["cls", "mask", "report", "qa_text", "bbox", "bbox_labels", "bbox_findings", "reg"]:
            if k in data_dicts[0]:
                available_keys.add(k)

        # Probe the native voxel spacing of the first sample (3-D only).
        # It feeds the "Native: ..." caption and the "Match native" and
        # "Half native" buttons under the spacing sliders.
        # The result is None on failure, for 2-D, or for an unsupported format.
        native_spacing = None
        if config.is_3d:
            # data_dicts[0]['img'] keeps the base_dir the user typed, with any "~".
            # os.path.isabs returns False for these paths. A join with base_dir
            # then gave "~/.../train/~/.../train/...". Expand both sides first,
            # so the rest uses real absolute paths.
            first_path = os.path.expanduser(data_dicts[0].get("img", ""))
            if first_path:
                if os.path.isabs(first_path):
                    full = first_path
                else:
                    full = os.path.join(os.path.expanduser(base_dir or ""), first_path)
                native_spacing = _probe_native_spacing(full)

        # get_data_dict builds cls and reg from the sorted columns that the table has (D-05).
        # The names must use the same columns in the same order.
        state = _make_state(
            data_dicts,
            [c for c in sorted(ds_obj.LABEL_COLS) if c in df.columns],
            [c for c in sorted(getattr(ds_obj, "REG_COLS", [])) if c in df.columns],
            available_keys,
            is_3d=config.is_3d,
            dataset_name=dataset_name,
            native_spacing=native_spacing,
        )
        progress(1.0)
        msg = f"Loaded {len(data_dicts)} samples.  Keys: {sorted(available_keys)}"
        msg += _dropped_text(missing)
        return state, msg, _format_native_spacing(native_spacing)
    except Exception as exc:
        return None, f"Error: {exc}", _format_native_spacing(None)


def _update_extra_fields(dataset_name: str):
    """Return gr.updates for base_dir, csv_path, both extra textboxes, the optional dropdown and BBox."""
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
    # Some entries (ROCO, MedPMC, ...) set only a placeholder. Show the field.
    label1 = label1 or ("Extra field" if ph1 else "")
    label2 = label2 or ("Extra field 2" if ph2 else "")
    dd_label = config.extra_dropdown_label if config else ""
    dd_choices = list(config.extra_dropdown_choices) if config else []
    dd_value = dd_choices[0] if dd_choices else None
    bbox = bool(config.bbox_default) if config else False
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
        gr.update(value=bbox),
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

    # A loaded state can hold "bbox_mask". The snippet does not list it, as in _run.
    keys_repr = "{" + ", ".join(f'"{k}"' for k in sorted(avail_keys - {"bbox_mask"})) + "}"
    lines.append("transform = (")
    if is_3d:
        hw_str = f"({hu_min}, {hu_max})" if hu_window else "None"
        base_transpose = not dataset_name.startswith("RAD-ChestCT")
        lines += [
            f"    {cls}(",
            f"        img_size={img_size},",
            f"        output_keys={keys_repr},",
            f"        pad={pad},",
            f"        hu_window={hw_str},",
        ]
        if not base_transpose:
            lines.append("        base_transpose=False,")
        lines.append("    )")
        if pixdim is not None:
            lines.append(f"    .with_spacing(({pixdim[0]}, {pixdim[1]}, {pixdim[2]}))")
    else:
        lines += [
            f"    {cls}(",
            f"        img_size={img_size},",
            f"        output_keys={keys_repr},",
            f"        pad={pad},",
        ]
        if _channels(dataset_name) == 3:
            lines.append("        channels=3,")
        lines.append("    )")
    if "bbox" in avail_keys:
        lines.append("    .with_bbox_as_mask()")

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
                    "    .with_elastic_deformation(",
                    f"        sigma_range=({p['elastic_sigma_min']}, {p['elastic_sigma_min']}),",
                    f"        magnitude_range=({p['elastic_mag_min']}, {p['elastic_mag_max']}),",
                    "        prob=0.5,",
                    "    )",
                ]
            else:
                lines += [
                    "    .with_elastic_deformation(",
                    f"        spacing={int(p['elastic_sigma_min'])},",
                    f"        magnitude_range=({p['elastic_mag_min']}, {p['elastic_mag_max']}),",
                    "        prob=0.5,",
                    "    )",
                ]
        elif aug == "affine":
            lines += [
                "    .with_affine(",
                f"        translate_range=({p['affine_translate']}, {p['affine_translate']}),",
                f"        rotate_range=({p['affine_rotate']},),",
                f"        scale_range=({p['affine_scale']}, {p['affine_scale']}),",
                f"        shear_range=({p['affine_shear']},),",
                "        prob=0.5,",
                "    )",
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
# several volume-sized float32 buffers and a deformation grid.  Two or more on
# a large volume can run out of memory with no traceback.  This budget makes
# the limit follow the free VRAM of the CUDA device at click time, not a fixed
# value.  Raise it if the lab GPU often flags safe combos as risky.
# Lower it if real out-of-memory errors pass the guard.
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
    bbox_classes=None,
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

    # Keep the full sample for "Run on same sample". The box filter applies to
    # this run only, so a box that the filter hides can come back.
    raw_sample = (
        reuse_sample if reuse_sample is not None else random.choice(data_dicts)
    )
    sample_dict = _filter_bbox_by_class(raw_sample, bbox_classes)
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
        # A copy, so the keys of the loaded state do not change.
        keys = set(avail_keys)
        if is_3d:
            hw = (hu_min, hu_max) if hu_window else None
            b = RadiologyTransform3D(
                img_size=img_size,
                output_keys=keys,
                pad=pad,
                hu_window=hw,
                base_transpose=not dataset_name.startswith("RAD-ChestCT"),
            )
            if pixdim is not None:
                b.with_spacing(pixdim)
        else:
            b = RadiologyTransform2D(img_size=img_size, output_keys=keys, pad=pad,
                                     channels=_channels(dataset_name))
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

    # Cache the transformed samples with the gallery. The "Next bbox →" button
    # can then show another bbox without a new transform run.
    # Keep only the keys that _sample_to_rgb reads, and only if the value is set.
    # Else the `"mask" in sample` check later finds None values.
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

    # Metadata outputs.  Classification (cls gives positive labels) and
    # regression (reg gives "col=value" pairs) share the Labels textbox.
    # If a dataset has both, the classification summary comes first and
    # the regression values follow on a new line.
    label_parts: list[str] = []
    if "cls" in base_sample and label_cols:
        vals = base_sample["cls"].float().tolist()
        pos = []
        for i, v in enumerate(vals):
            if not v > 0.5:
                continue
            grade = int(round(v))
            pos.append(f"{label_cols[i]} ({grade})" if grade > 1 else label_cols[i])
        # "No finding" is a label name in many datasets, so do not use it here.
        label_parts.append(", ".join(pos) if pos else "(no positive label)")
    if "reg" in base_sample and reg_cols:
        vals = base_sample["reg"].float().tolist()
        label_parts.append(
            ", ".join(f"{reg_cols[i]}={v:.3f}" for i, v in enumerate(vals))
        )
    labels_text = "\n".join(label_parts)

    # For bbox-only datasets (no cls), show the per-box labels in the Labels panel.
    # Chest ImaGenome shows "anatomy: findings" for each region.
    # Other bbox sets (no findings column) show the list of box class labels.
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
        raw_sample,
        gallery_samples,
    )


def run_2d(ds_state, img_size, pad, show_mask, show_bbox, bbox_classes, *aug_inputs):
    results = _run(ds_state, img_size, pad, show_mask, show_bbox, *aug_inputs,
                   bbox_classes=bbox_classes)
    # results: gallery, info, image_stats, labels, report, qa_text, code, sample_dict, gallery_samples
    return *results, gr.update(interactive=results[7] is not None)


def run_2d_same(
    ds_state, last_sample, img_size, pad, show_mask, show_bbox, bbox_classes,
    *aug_inputs
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
        bbox_classes=bbox_classes,
    )
    return results  # 8 values, no need to update button


def run_3d(
    ds_state,
    img_size,
    pad,
    show_mask,
    show_bbox,
    bbox_classes,
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
        bbox_classes=bbox_classes,
    )
    return *results, gr.update(interactive=results[7] is not None)


def run_3d_same(
    ds_state,
    last_sample,
    img_size,
    pad,
    show_mask,
    show_bbox,
    bbox_classes,
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
        bbox_classes=bbox_classes,
    )


# ---------------------------------------------------------------------------
# Next-bbox cycling. Cervical and lumbar spine only.
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
    """Re-render the gallery with the next bbox in focus.

    Pure numpy work. No new transform run and no new augmentation draw."""
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
    """Render augmentation checkboxes and sliders. Return a tuple of components.

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

        # Power-user switch for the heavy-augmentation VRAM budget.  Off by
        # default.  When on, a budget violation gives a warning toast and
        # does not block the run.  Only the 3-D tab has it, because the
        # guard is 3-D only.
        if is_3d:
            with gr.Group():
                with gr.Row():
                    # No info= text or caption, like the other checkboxes in this
                    # accordion (label only).  "(advanced)" marks a power-user switch.
                    # The OOM error toast says when to turn it on.
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
        # Same labels and visibility as a later change of the variant.
        _, _, _ef1, _ef2, _, _bbox = _update_extra_fields(first_variant)
        extra_field = gr.Textbox(
            label=_ef1["label"],
            placeholder=_first.extra_placeholder,
            visible=_ef1["visible"],
        )
        extra_field2 = gr.Textbox(
            label=_ef2["label"],
            placeholder=_first.extra_placeholder2,
            visible=_ef2["visible"],
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
                want_bbox = gr.Checkbox(label="BBox", value=_bbox["value"])
                # Regression targets (continuous). A dataset with no regression
                # target warns and gives no "reg" key (D-11).
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
            return (gr.update(),) * 7
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
            want_bbox,
        ],
        queue=False,
        show_progress="hidden",
    )
    # Variant change: refresh placeholders/labels for the newly selected variant.
    variant_dd.change(
        fn=_update_extra_fields,
        inputs=[variant_dd],
        outputs=[base_dir, csv_path, extra_field, extra_field2, extra_dropdown, want_bbox],
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
    """Render the full tab body of one modality: load panel, controls, runner, outputs.

    All datasets within a modality must currently share the same ``is_3d``
    dimension.  Mixed-dimension modalities are not supported yet.
    They would need controls that change per dataset (3-D controls would
    appear and disappear as the dropdown changes). That is a bigger UI change.
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

            # HU (Hounsfield Units) only make sense for CT. MRI intensities depend
            # on the scanner and sequence, so hide the windowing controls and
            # turn them off on MRI tabs.  They stay as real (hidden) components,
            # so the run_3d wiring is the same for all 3-D modalities.
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
                    # Voxel spacing (mm).  On by default, so anisotropic MRI
                    # (for example, 0.5 x 0.5 x 4 mm lumbar spine) does not collapse
                    # along z under ResizeD(size_mode="longest").  Per-axis
                    # controls let users match the native spacing of the dataset.
                    # Turn it off to see the raw geometry, or when an earlier
                    # step already resampled.
                    gr.Markdown("**Voxel spacing (mm)**")
                    iso_spacing = gr.Checkbox(label="Resample", value=True)
                    # Native spacing readout. It sits above the sliders, so the
                    # user sees the dataset's start values before any change.
                    # load_dataset fills it.
                    native_spacing_md = gr.Markdown(
                        "Native spacing: _load a dataset to see_",
                        elem_classes=["native-spacing-readout"],
                    )
                    with gr.Row():
                        pixdim_x = gr.Slider(
                            _SPACING_MIN, _SPACING_MAX, value=1.0, step=0.05, label="x (mm)"
                        )
                        pixdim_y = gr.Slider(
                            _SPACING_MIN, _SPACING_MAX, value=1.0, step=0.05, label="y (mm)"
                        )
                        pixdim_z = gr.Slider(
                            _SPACING_MIN, _SPACING_MAX, value=1.0, step=0.05, label="z (mm)"
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
                bbox_classes = gr.Dropdown(
                    label="BBox classes  (empty = show all)",
                    choices=[],
                    value=[],
                    multiselect=True,
                    interactive=True,
                    visible=False,
                )

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
        # Cervical and lumbar only: go to the next bbox without a new transform run.
        # Always hidden on 2-D tabs.  On 3-D tabs it starts hidden and shows only
        # after a focus-first sample with bboxes is rendered.
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
        # load_dataset always returns 3 values.  3-D tabs send the third
        # (native-spacing string) to the markdown caption.  2-D tabs send it
        # to a hidden gr.State. Else Gradio warns "too many return values"
        # and prints the full state dict to the terminal.
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
    # Box-class vocabulary is only knowable after a dataset is loaded, so the
    # filter is populated (and revealed) here rather than at build time.  Hidden
    # entirely for datasets that ship no box labels.
    load_event.then(
        fn=lambda st: gr.update(
            choices=_bbox_class_vocab(st),
            value=[],
            visible=bool(_bbox_class_vocab(st)),
        ),
        inputs=[state],
        outputs=[bbox_classes],
        queue=False,
    )
    stop_btn.click(
        fn=lambda: (gr.update(visible=True), gr.update(visible=False)),
        outputs=[load_btn, stop_btn],
        cancels=[load_event],
        queue=False,
    )

    # run_3d and run_3d_same take the extra HU args. The run_2d versions do not.
    run_fn = run_3d if is_3d else run_2d
    run_same_fn = run_3d_same if is_3d else run_2d_same
    extra_3d_inputs = (
        [hu_window, hu_min, hu_max, iso_spacing, pixdim_x, pixdim_y, pixdim_z]
        if is_3d
        else []
    )

    run_btn.click(
        fn=run_fn,
        inputs=[state, img_size, pad, show_mask, show_bbox, bbox_classes]
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
        inputs=[state, last_sample, img_size, pad, show_mask, show_bbox, bbox_classes]
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
        # A re-run on the same sample keeps the bbox set. Only check the button state again.
        fn=_next_btn_interactive,
        inputs=[state, gallery_samples_state],
        outputs=[next_bbox_btn],
    )

    next_bbox_btn.click(
        fn=_on_next_bbox,
        inputs=[gallery_samples_state, bbox_idx_state, show_mask, show_bbox, state],
        outputs=[gallery, bbox_idx_state],
    )

    # Reset the bbox cursor when the user turns bbox visibility off or on,
    # or changes the dataset variant. The next click starts at index 0.
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
        # Preset buttons only set the slider values.  A new render happens only
        # when the user clicks "Run on same sample". Spacing edits do not
        # start the pipeline.
        def _preset_1mm():
            return gr.update(value=1.0), gr.update(value=1.0), gr.update(value=1.0)

        def _preset_match(s):
            if s and s.get("native_spacing"):
                return tuple(gr.update(value=v) for v in _match_native_values(s["native_spacing"]))
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

# Gradio 6 takes css in launch(). Blocks() ignores it.
with gr.Blocks(title="RadHarmony Visualizer") as demo:
    gr.Markdown("# RadHarmony — Dataset & Transform Visualizer")
    gr.Markdown(
        "**Workflow:** &nbsp; ① Pick a dataset and click **Load** &nbsp;→&nbsp; "
        "② Adjust preprocessing & augmentations &nbsp;→&nbsp; "
        "Click **Run on random sample** to inspect each effect side-by-side."
    )

    # ── One tab per modality ─────────────────────────────────────────────────
    # SUPPORTED_MODALITIES sets the display order.
    # Modalities with no registered datasets are skipped.
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
    demo.launch(server_name="0.0.0.0", server_port=_args.server_port, css=_CSS)
