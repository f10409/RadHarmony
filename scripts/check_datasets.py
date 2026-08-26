#!/usr/bin/env python
"""Headless dataset integrity check — the agent / CI counterpart to
``notebooks/datasets/dataset_integrity_check.ipynb``.

For each dataset it harmonizes + instantiates the class, sub-samples a handful
of rows, iterates every sample through a ``DataLoader`` and validates the image
shape / dtype / range plus the cls / mask / bbox / report outputs.  It prints a
per-dataset report and a final summary table, optionally writes a JSON array,
and exits non-zero if any dataset ERRORed — so an agent (or CI) can gate on it.

    uv run python scripts/check_datasets.py --list
    uv run python scripts/check_datasets.py --datasets chexpert_train,vindr_train
    uv run python scripts/check_datasets.py --max-samples 20 --json report.json
    uv run python scripts/check_datasets.py            # every dataset

The notebook is better for a *human* (it renders images); this script is
better for an *agent* (deterministic JSON + exit code).  Both share the check
logic in :mod:`radharmony.integrity`, so the two never drift.

Paths below point at this environment's NAS layout — edit them, or comment out
any dataset you don't have, to match yours.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field

from radharmony.dataset import (
    BRAXDataset,
    BRAXPNGDataset,
    ChestXray14TrainDataset,
    ChestXray14TestDataset,
    ChestXray14BboxDataset,
    CheXpertTrainDataset,
    CheXpertValidDataset,
    CheXpertPlusDataset,
    CheXlocalizeDataset,
    VQARadDataset,
    MIMICCXRDataset,
    MIMICCXRJPGDataset,
    MIMICCXRJPGTestDataset,
    OpenICXRDataset,
    PadChestDataset,
    ReXGradientTrainDataset,
    ReXGradientValidDataset,
    ReXGradientTestDataset,
    RANZCRClipDataset,
    VinDrCXRTrainDataset,
    VinDrCXRTestDataset,
    ChestImaGenomeGoldDataset,
    ChestImaGenomeSilverDataset,
    SIIMACRPTXTrainDataset,
    SIIMCOVID19TrainDataset,
    RSNAPneumoniaKaggleTrainDataset,
    MontgomeryCXRDataset,
    ShenzhenCXRDataset,
    TAIXRay512Dataset,
    RSNAPEDetectionTrainDataset,
    RSNABoneAgeTrainDataset,
    RSNA2022CervicalSpineTrainDataset,
    RSNA2022CervicalSpineBboxDataset,
    RSNA2024LumbarSpineTrainDataset,
    RSNAAbdominalTrauma2023Dataset,
    CTRATEDataset,
    RadChestCTDataset,
)
from radharmony.integrity import check_dataset, print_report, subsample_dataset

# ── Dataset paths (local NAS) ── explicit; no auto-discovery ───────────────
# CheXpert
CHEXPERT_TRAIN_BASE   = "/mnt/NAS3/datasets/external/CheXpert-v1.0/train/"
CHEXPERT_TRAIN_CSV    = "/mnt/NAS3/datasets/external/CheXpert-v1.0/train.csv"
CHEXPERT_VALID_BASE   = "/mnt/NAS3/datasets/external/CheXpert-v1.0/valid/"
CHEXPERT_VALID_CSV    = "/mnt/NAS3/datasets/external/CheXpert-v1.0/valid.csv"
CHEXPERT_PLUS_BASE    = "/mnt/NAS4/datasets/external/CheXpert_Plus/chexpertplus/DICOM/Uncompressed/"
CHEXPERT_PLUS_CSV     = "/mnt/NAS4/datasets/external/CheXpert_Plus/chexpertplus/df_chexpert_plus_240401.csv"
CHEXPERT_PLUS_JSON    = "/mnt/NAS4/datasets/external/CheXpert_Plus/chexpertplus/report_fixed.json"

# CheXlocalize (CheXpert "test" split + per-pathology segmentation masks)
CHEXLOCALIZE_BASE      = "/mnt/NAS4/datasets/external/chexlocalize/CheXpert/test/"
CHEXLOCALIZE_CSV       = "/mnt/NAS4/datasets/external/chexlocalize/CheXpert/test_labels.csv"
CHEXLOCALIZE_MASK_JSON = "/mnt/NAS4/datasets/external/chexlocalize/CheXlocalize/gt_segmentations_test.json"

# VQA-RAD (visual question answering)
VQA_RAD_BASE = "/mnt/NAS4/datasets/external/VQA-RAD/VQA_RAD Image Folder"

# MIMIC-CXR (DICOM + JPG)
MIMIC_CXR_BASE        = "/mnt/NAS4/datasets/external/MIMIC-CXR-V2-AWS/files/"
MIMIC_RECORD_CSV      = "/mnt/NAS4/datasets/external/MIMIC-CXR-V2-AWS/cxr-record-list.csv.gz"
MIMIC_STUDY_CSV       = "/mnt/NAS4/datasets/external/MIMIC-CXR-V2-AWS/cxr-study-list.csv.gz"
MIMIC_CXR_JPG_BASE    = "/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files/"
MIMIC_META_CSV        = "/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv"
MIMIC_CHEXPERT_CSV    = "/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv"
MIMIC_TEST_LABEL_CSV  = "/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.1.0-test-set-labeled.csv"

# ChestX-ray14
CHESTXRAY14_BASE      = "/mnt/NAS3/datasets/external/NIH_CXR/CXR14/"
CHESTXRAY14_CSV       = "/mnt/NAS3/datasets/external/NIH_CXR/CXR14/Data_Entry_2017.csv"
CHESTXRAY14_BBOX_CSV  = "/mnt/NAS3/datasets/external/NIH_CXR/CXR14/BBox_List_2017.csv"

# PadChest
PADCHEST_BASE         = "/mnt/NAS4/datasets/external/PadChest/images/"
PADCHEST_CSV          = "/mnt/NAS4/datasets/external/PadChest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"

# BRAX
BRAX_BASE             = "/mnt/NAS4/datasets/external/BRAX/"
BRAX_CSV              = "/mnt/NAS4/datasets/external/BRAX/master_spreadsheet_update.csv"

# ReXGradient-160K (per-split metadata JSON)
REXGRADIENT_BASE      = "/mnt/NAS4/datasets/external/ReXGradient-160K/deid_png/"
REXGRADIENT_VALID_JSON = "/mnt/NAS4/datasets/external/ReXGradient-160K/download/metadata/valid_metadata_view_position.json"
REXGRADIENT_TEST_JSON  = "/mnt/NAS4/datasets/external/ReXGradient-160K/download/metadata/test_metadata_view_position.json"
REXGRADIENT_TRAIN_JSON = "/mnt/NAS4/datasets/external/ReXGradient-160K/download/metadata/train_metadata_view_position.json"

# OpenI (label + reports parsed from XML under base dir; no csv arg)
OPENI_BASE            = "/mnt/NAS4/datasets/external/OpenI-IU-CXR/"

# RANZCR CLiP
RANZCR_CLIP_BASE      = "/mnt/NAS4/datasets/external/ranzcr-clip-catheter-line-classification/"
RANZCR_CSV            = "/mnt/NAS4/datasets/external/ranzcr-clip-catheter-line-classification/train.csv"
RANZCR_ANNOT_CSV      = "/mnt/NAS4/datasets/external/ranzcr-clip-catheter-line-classification/train_annotations.csv"

# VinDr-CXR
VINDR_CXR_TRAIN_BASE  = "/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/train/"
VINDR_TRAIN_LABEL_CSV = "/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_train.csv"
VINDR_TRAIN_BBOX_CSV  = "/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_train.csv"
VINDR_CXR_TEST_BASE   = "/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/test/"
VINDR_TEST_LABEL_CSV  = "/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv"
VINDR_TEST_BBOX_CSV   = "/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_test.csv"

# Chest ImaGenome (annotation layer over MIMIC-CXR DICOM; PhysioNet-credentialed)
CHEST_IMAGENOME_ANN   = "/mnt/NAS4/datasets/external/CHEST-IMAGENOME"
MIMIC_CXR_DICOM_BASE  = "/mnt/NAS4/datasets/external/MIMIC-CXR-V2-AWS/files/"

# SIIM-ACR Pneumothorax
SIIM_ACR_BASE         = "/mnt/NAS3/datasets/external/SIIM_ACR_Pneumothorax/dicom-images-train/"
SIIM_ACR_CSV          = "/mnt/NAS3/datasets/external/SIIM_ACR_Pneumothorax/train-rle.csv"

# SIIM COVID-19
SIIM_COVID19_BASE     = "/mnt/NAS4/datasets/external/siim-covid19-detection/train/"
SIIM_COVID_STUDY_CSV  = "/mnt/NAS4/datasets/external/siim-covid19-detection/train_study_level.csv"
SIIM_COVID_IMAGE_CSV  = "/mnt/NAS4/datasets/external/siim-covid19-detection/train_image_level.csv"

# RSNA Pneumonia (Kaggle)
RSNA_PNEUMONIA_BASE   = "/mnt/NAS4/datasets/external/rsna-pneumonia-detection-challenge/stage_2_train_images/"
RSNA_PNEU_CSV         = "/mnt/NAS4/datasets/external/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv"
RSNA_PNEU_BBOX_CSV    = "/mnt/NAS4/datasets/external/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv"

# TAIX-Ray (512px)
TAIX_RAY_512_BASE     = "/mnt/NAS4/datasets/external/TAIX-Ray/data_512/images/"
TAIX_RAY_512_CSV      = "/mnt/NAS4/datasets/external/TAIX-Ray/data_512/annotation.csv"

# Montgomery / Shenzhen (labels in filename/siblings; no csv arg)
MONTGOMERY_CXR_BASE   = "/mnt/NAS4/datasets/external/Montgomery-CXR/MontgomerySet/CXR_png/"
SHENZHEN_CXR_BASE     = "/mnt/NAS4/datasets/external/Shenzhen-Hospital-CXR-Set/CXR_png/"

# RSNA PE Detection (3-D)
RSNA_PE_DETECTION_BASE = "/mnt/NAS4/datasets/external/rsna_pe_dataset/train/"
RSNA_PE_CSV            = "/mnt/NAS4/datasets/external/rsna_pe_dataset/train.csv"

# RSNA Pediatric Bone Age
RSNA_BONE_AGE_TRAIN_BASE = "/mnt/NAS4/datasets/external/rsna-boneage/boneage-training-dataset/"
RSNA_BONE_AGE_TRAIN_CSV  = "/mnt/NAS4/datasets/external/rsna-boneage/train.csv"

# RSNA 2022 Cervical Spine (3-D)
RSNA_2022_CERVICAL_BASE     = "/mnt/NAS4/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_images/"
RSNA_2022_CERVICAL_CSV      = "/mnt/NAS4/datasets/external/rsna-2022-cervical-spine-fracture-detection/train.csv"
RSNA_2022_CERVICAL_SEG_DIR  = "/mnt/NAS4/datasets/external/rsna-2022-cervical-spine-fracture-detection/segmentations/"
RSNA_2022_CERVICAL_BBOX_CSV = "/mnt/NAS4/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv"

# RSNA 2023 Abdominal Trauma (3-D)
RSNA_2023_ABDOMINAL_BASE       = "/mnt/NAS4/datasets/external/rsna-2023-abdominal-trauma-detection/train_images/"
RSNA_2023_ABDOMINAL_CSV        = "/mnt/NAS4/datasets/external/rsna-2023-abdominal-trauma-detection/train_2024.csv"
RSNA_2023_ABDOMINAL_SERIES_CSV = "/mnt/NAS4/datasets/external/rsna-2023-abdominal-trauma-detection/train_series_meta.csv"

# RSNA 2024 Lumbar Spine (3-D MRI)
RSNA_2024_LUMBAR_BASE       = "/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train_images/"
RSNA_2024_LUMBAR_CSV        = "/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train.csv"
RSNA_2024_LUMBAR_SERIES_CSV = "/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv"
RSNA_2024_LUMBAR_COORD_CSV  = "/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv"

# CT-RATE (3-D NIfTI)
CTRATE_BASE           = "/mnt/NAS4/datasets/external/CT-RATE/dataset/train_fixed/"
CTRATE_CSV            = "/mnt/NAS4/datasets/external/CT-RATE/dataset/tables/train_predicted_labels.csv"
CTRATE_VIEW_CSV       = "/mnt/NAS4/datasets/external/CT-RATE/dataset/tables/train_metadata.csv"

# RAD-ChestCT (3-D .npz)
RADCHESTCT_BASE       = "/mnt/NAS4/datasets/external/RAD-ChestCT/images/"
RADCHESTCT_CSV        = "/mnt/NAS4/datasets/external/RAD-ChestCT/tables/CT_Scan_Metadata_Complete_35747.csv"
RADCHESTCT_LABEL_CSV  = "/mnt/NAS4/datasets/external/RAD-ChestCT/tables/imgtrain_Abnormality_and_Location_Labels.csv"
RADCHESTCT_BBOX_CSV   = "/mnt/NAS4/datasets/external/RAD-ChestCT/tables/Extrema_35747.csv"

# Scratch dir for datasets that rasterize/decode masks to PNG on the fly.
MASK_OUT = os.path.expanduser("~/.cache/radharmony/verify_masks")


@dataclass
class DatasetSpec:
    """One dataset to check: a registry key, the class, its constructor
    kwargs, and an optional DataLoader-worker override (3-D volumes use fewer
    workers to bound memory)."""

    key: str
    cls: type
    kwargs: dict = field(default_factory=dict)
    workers: int | None = None


def get_specs() -> list[DatasetSpec]:
    """The public (external) dataset registry.  Mirrors the notebook's
    ``build(...)`` calls one-to-one."""
    S = DatasetSpec
    return [
        # ── VQA ──
        S("vqa_rad", VQARadDataset, dict(
            base_image_dir=VQA_RAD_BASE, output_question_type=True)),
        # ── 2-D CXR ──
        S("chexpert_train", CheXpertTrainDataset, dict(
            base_image_dir=CHEXPERT_TRAIN_BASE, csv_path=CHEXPERT_TRAIN_CSV,
            output_cls=True, drop_uncertain=False)),
        S("chexpert_valid", CheXpertValidDataset, dict(
            base_image_dir=CHEXPERT_VALID_BASE, csv_path=CHEXPERT_VALID_CSV,
            output_cls=True)),
        S("chexpert_plus", CheXpertPlusDataset, dict(
            base_image_dir=CHEXPERT_PLUS_BASE, csv_path=CHEXPERT_PLUS_CSV,
            label_json_path=CHEXPERT_PLUS_JSON, output_cls=True,
            output_report=True, drop_uncertain=False)),
        S("chexlocalize", CheXlocalizeDataset, dict(
            base_image_dir=CHEXLOCALIZE_BASE, csv_path=CHEXLOCALIZE_CSV,
            output_cls=True)),
        S("chexlocalize_mask", CheXlocalizeDataset, dict(
            base_image_dir=CHEXLOCALIZE_BASE, csv_path=CHEXLOCALIZE_CSV,
            mask_json_path=CHEXLOCALIZE_MASK_JSON,
            mask_output_dir=os.path.join(MASK_OUT, "chexlocalize"),
            output_cls=True, output_mask=True)),
        S("mimic_cxr", MIMICCXRDataset, dict(
            base_image_dir=MIMIC_CXR_BASE, csv_path=MIMIC_RECORD_CSV,
            label_csv_path=MIMIC_CHEXPERT_CSV, report_csv_path=MIMIC_STUDY_CSV,
            output_cls=True, output_report=True, drop_uncertain=False)),
        S("mimic_cxr_jpg", MIMICCXRJPGDataset, dict(
            base_image_dir=MIMIC_CXR_JPG_BASE, csv_path=MIMIC_META_CSV,
            label_csv_path=MIMIC_CHEXPERT_CSV, output_cls=True)),
        S("mimic_cxr_jpg_test", MIMICCXRJPGTestDataset, dict(
            base_image_dir=MIMIC_CXR_JPG_BASE, csv_path=MIMIC_META_CSV,
            label_csv_path=MIMIC_TEST_LABEL_CSV, output_cls=True)),
        S("chestxray14_train", ChestXray14TrainDataset, dict(
            base_image_dir=CHESTXRAY14_BASE, csv_path=CHESTXRAY14_CSV,
            bbox_csv_path=CHESTXRAY14_BBOX_CSV, output_cls=True)),
        S("chestxray14_test", ChestXray14TestDataset, dict(
            base_image_dir=CHESTXRAY14_BASE, csv_path=CHESTXRAY14_CSV,
            bbox_csv_path=CHESTXRAY14_BBOX_CSV, output_cls=True)),
        S("chestxray14_bbox", ChestXray14BboxDataset, dict(
            base_image_dir=CHESTXRAY14_BASE, csv_path=CHESTXRAY14_CSV,
            bbox_csv_path=CHESTXRAY14_BBOX_CSV, output_cls=True, output_bbox=True)),
        S("padchest", PadChestDataset, dict(
            base_image_dir=PADCHEST_BASE, csv_path=PADCHEST_CSV,
            output_cls=True, output_report=True)),
        S("brax", BRAXDataset, dict(
            base_image_dir=BRAX_BASE, csv_path=BRAX_CSV,
            uncertain_strategy="u_zeros", output_cls=True)),
        S("brax_png", BRAXPNGDataset, dict(
            base_image_dir=BRAX_BASE, csv_path=BRAX_CSV,
            uncertain_strategy="u_zeros", output_cls=True)),
        S("openi_cxr", OpenICXRDataset, dict(
            base_image_dir=OPENI_BASE, output_cls=True, output_report=True)),
        S("rexgradient_valid", ReXGradientValidDataset, dict(
            base_image_dir=REXGRADIENT_BASE, csv_path=REXGRADIENT_VALID_JSON,
            output_report=True)),
        S("rexgradient_test", ReXGradientTestDataset, dict(
            base_image_dir=REXGRADIENT_BASE, csv_path=REXGRADIENT_TEST_JSON,
            output_report=True)),
        S("rexgradient_train", ReXGradientTrainDataset, dict(
            base_image_dir=REXGRADIENT_BASE, csv_path=REXGRADIENT_TRAIN_JSON,
            output_report=True)),
        S("vindr_train", VinDrCXRTrainDataset, dict(
            base_image_dir=VINDR_CXR_TRAIN_BASE, csv_path=VINDR_TRAIN_LABEL_CSV,
            bbox_csv_path=VINDR_TRAIN_BBOX_CSV, output_cls=True, output_bbox=True)),
        S("vindr_test", VinDrCXRTestDataset, dict(
            base_image_dir=VINDR_CXR_TEST_BASE, csv_path=VINDR_TEST_LABEL_CSV,
            bbox_csv_path=VINDR_TEST_BBOX_CSV, output_cls=True, output_bbox=True)),
        S("chest_imagenome_gold", ChestImaGenomeGoldDataset, dict(
            base_image_dir=MIMIC_CXR_DICOM_BASE, annotation_dir=CHEST_IMAGENOME_ANN,
            output_bbox=True)),
        # silver = full ~240k-image release; harmonize reads one DICOM header per
        # image, so a full run is slow. "valid" is the smallest official split.
        S("chest_imagenome_silver", ChestImaGenomeSilverDataset, dict(
            base_image_dir=MIMIC_CXR_DICOM_BASE, annotation_dir=CHEST_IMAGENOME_ANN,
            split="valid", output_bbox=True)),
        S("siim_acr_ptx", SIIMACRPTXTrainDataset, dict(
            base_image_dir=SIIM_ACR_BASE, csv_path=SIIM_ACR_CSV,
            mask_output_dir=os.path.join(MASK_OUT, "siim_acr"),
            output_cls=True, output_mask=True)),
        S("siim_covid19", SIIMCOVID19TrainDataset, dict(
            base_image_dir=SIIM_COVID19_BASE, csv_path=SIIM_COVID_STUDY_CSV,
            image_csv_path=SIIM_COVID_IMAGE_CSV, output_cls=True, output_bbox=True)),
        S("rsna_pneumonia", RSNAPneumoniaKaggleTrainDataset, dict(
            base_image_dir=RSNA_PNEUMONIA_BASE, csv_path=RSNA_PNEU_CSV,
            bbox_csv_path=RSNA_PNEU_BBOX_CSV, output_cls=True, output_bbox=True)),
        S("ranzcr", RANZCRClipDataset, dict(
            base_image_dir=RANZCR_CLIP_BASE, csv_path=RANZCR_CSV,
            include_test_split=False, output_cls=True)),
        S("ranzcr_mask", RANZCRClipDataset, dict(
            base_image_dir=RANZCR_CLIP_BASE, csv_path=RANZCR_CSV,
            annotations_csv_path=RANZCR_ANNOT_CSV,
            mask_output_dir=os.path.join(MASK_OUT, "ranzcr"),
            include_test_split=False, output_cls=True, output_mask=True)),
        S("taix_ray_512", TAIXRay512Dataset, dict(
            base_image_dir=TAIX_RAY_512_BASE, csv_path=TAIX_RAY_512_CSV,
            label_mode="ordinal", output_cls=True)),
        S("montgomery", MontgomeryCXRDataset, dict(
            base_image_dir=MONTGOMERY_CXR_BASE,
            mask_output_dir=os.path.join(MASK_OUT, "montgomery"),
            output_cls=True, output_mask=True, output_report=True)),
        S("shenzhen", ShenzhenCXRDataset, dict(
            base_image_dir=SHENZHEN_CXR_BASE,
            mask_output_dir=os.path.join(MASK_OUT, "shenzhen"),
            output_cls=True, output_mask=True, output_report=True)),
        S("rsna_bone_age", RSNABoneAgeTrainDataset, dict(
            base_image_dir=RSNA_BONE_AGE_TRAIN_BASE, csv_path=RSNA_BONE_AGE_TRAIN_CSV,
            output_reg=True)),
        # ── 3-D (fewer workers to bound memory) ──
        S("rsna_pe", RSNAPEDetectionTrainDataset, dict(
            base_image_dir=RSNA_PE_DETECTION_BASE, csv_path=RSNA_PE_CSV,
            output_cls=True), workers=2),
        S("rsna_cervical", RSNA2022CervicalSpineTrainDataset, dict(
            base_image_dir=RSNA_2022_CERVICAL_BASE, csv_path=RSNA_2022_CERVICAL_CSV,
            segmentation_dir=RSNA_2022_CERVICAL_SEG_DIR,
            output_cls=True, output_mask=True), workers=2),
        S("rsna_cervical_bbox", RSNA2022CervicalSpineBboxDataset, dict(
            base_image_dir=RSNA_2022_CERVICAL_BASE, csv_path=RSNA_2022_CERVICAL_CSV,
            bbox_csv_path=RSNA_2022_CERVICAL_BBOX_CSV,
            output_cls=True, output_bbox=True), workers=2),
        S("rsna_lumbar", RSNA2024LumbarSpineTrainDataset, dict(
            base_image_dir=RSNA_2024_LUMBAR_BASE, csv_path=RSNA_2024_LUMBAR_CSV,
            series_description_csv_path=RSNA_2024_LUMBAR_SERIES_CSV,
            coord_csv_path=RSNA_2024_LUMBAR_COORD_CSV,
            output_cls=True, output_bbox=True), workers=2),
        S("rsna_abdominal", RSNAAbdominalTrauma2023Dataset, dict(
            base_image_dir=RSNA_2023_ABDOMINAL_BASE, csv_path=RSNA_2023_ABDOMINAL_CSV,
            series_meta_csv_path=RSNA_2023_ABDOMINAL_SERIES_CSV,
            output_cls=True), workers=2),
        S("ct_rate", CTRATEDataset, dict(
            base_image_dir=CTRATE_BASE, csv_path=CTRATE_CSV,
            view_position_csv_path=CTRATE_VIEW_CSV, output_cls=True), workers=2),
        S("rad_chest_ct", RadChestCTDataset, dict(
            base_image_dir=RADCHESTCT_BASE, csv_path=RADCHESTCT_CSV,
            label_csv_path=RADCHESTCT_LABEL_CSV, bbox_csv_path=RADCHESTCT_BBOX_CSV,
            output_cls=True, output_bbox=True), workers=2),
    ]


def run_one(spec: DatasetSpec, max_samples: int | None, default_workers: int):
    """Instantiate -> sub-sample -> integrity-check a single dataset."""
    print("\n" + "#" * 72)
    print(f"# {spec.key}  ({spec.cls.__name__})")
    print("#" * 72)
    try:
        ds = spec.cls(cache_dir=None, **spec.kwargs)
    except Exception as e:
        traceback.print_exc()
        from radharmony.integrity import IntegrityReport
        return IntegrityReport(
            dataset=spec.cls.__name__, status="ERROR",
            error_message=f"__init__ failed: {e}")
    try:
        n0, n1 = subsample_dataset(ds, max_samples)
        if n1 < n0:
            print(f"  sub-sampled {n0:,} -> {n1} rows")
    except Exception as e:
        print(f"  [warn] could not sub-sample: {type(e).__name__}: {e}")
    workers = spec.workers if spec.workers is not None else default_workers
    return check_dataset(ds, num_workers=workers)


def print_summary(rows: list[tuple[str, "IntegrityReport"]]) -> None:
    """Final aligned status table across all datasets."""
    icon = {"OK": "✅", "WARN": "⚠️", "ERROR": "❌"}
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'key':22s} {'status':7s} {'ok/total':>14s} {'img_range':>18s}  time")
    print("-" * 78)
    for key, r in rows:
        rng = f"[{r.img_min:.2f},{r.img_max:.2f}]" if r.success else "-"
        ok = f"{r.success:,}/{r.total_samples:,}"
        print(f"{key:22s} {icon.get(r.status,'?')+r.status:7s} {ok:>14s} "
              f"{rng:>18s}  {r.total_time_s:.1f}s")


def main(extra_specs: list[DatasetSpec] | None = None, argv=None) -> int:
    specs = get_specs() + (extra_specs or [])
    by_key = {s.key: s for s in specs}

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", help="comma-separated keys to run (default: all)")
    p.add_argument("--skip", help="comma-separated keys to skip")
    p.add_argument("--max-samples", type=int, default=60,
                   help="rows to sub-sample per dataset (0 = every sample)")
    p.add_argument("--num-workers", type=int, default=8,
                   help="default DataLoader workers (3-D specs may override)")
    p.add_argument("--json", metavar="PATH",
                   help="write JSON report array to PATH ('-' for stdout)")
    p.add_argument("--list", action="store_true",
                   help="list available dataset keys and exit")
    args = p.parse_args(argv)

    if args.list:
        for s in specs:
            print(f"  {s.key:22s} {s.cls.__name__}")
        return 0

    selected = list(specs)
    if args.datasets:
        want = [k.strip() for k in args.datasets.split(",") if k.strip()]
        unknown = [k for k in want if k not in by_key]
        if unknown:
            p.error(f"unknown dataset key(s): {', '.join(unknown)}\n"
                    f"run with --list to see valid keys")
        selected = [by_key[k] for k in want]
    if args.skip:
        drop = {k.strip() for k in args.skip.split(",")}
        selected = [s for s in selected if s.key not in drop]

    os.makedirs(MASK_OUT, exist_ok=True)
    max_samples = args.max_samples or None

    # With --json -, JSON must be the ONLY thing on stdout, so route the
    # human-readable report/summary to stderr and keep stdout pure JSON.
    json_to_stdout = args.json == "-"
    human = sys.stderr if json_to_stdout else sys.stdout

    rows, json_out = [], []
    with contextlib.redirect_stdout(human):
        for spec in selected:
            r = run_one(spec, max_samples, args.num_workers)
            label_cols = list(getattr(spec.cls, "LABEL_COLS", []) or [])
            print_report(r, label_cols)
            rows.append((spec.key, r))
            d = r.to_dict(label_cols)
            d["key"] = spec.key
            json_out.append(d)

        print_summary(rows)
        n_error = sum(1 for _, r in rows if r.status == "ERROR")
        n_warn = sum(1 for _, r in rows if r.status == "WARN")
        print(f"\n{len(rows)} dataset(s): "
              f"{len(rows) - n_error - n_warn} OK, {n_warn} WARN, {n_error} ERROR")

    if args.json:
        payload = json.dumps(json_out, indent=2)
        if json_to_stdout:
            print(payload)  # real stdout — pure JSON
        else:
            with open(args.json, "w") as f:
                f.write(payload)
            print(f"\nWrote JSON report -> {args.json}")

    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
