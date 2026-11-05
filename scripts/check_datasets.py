#!/usr/bin/env python
"""Headless dataset integrity check. It is the agent and CI counterpart to
``notebooks/datasets/dataset_integrity_check.ipynb``.

For each dataset it harmonizes and creates the class, then takes a few rows.
It loads every sample through a ``DataLoader``. It checks the image shape,
dtype and range, and the cls, mask, bbox and report outputs.  It prints a report for
each dataset and a final summary table. It can write a JSON array. It exits
with a non-zero code if any dataset has an ERROR, so an agent or CI can use it.

Usage::

    uv run python scripts/check_datasets.py --list
    uv run python scripts/check_datasets.py --datasets chexpert_train,vindr_cxr_train
    uv run python scripts/check_datasets.py --max-samples 20 --json report.json
    uv run python scripts/check_datasets.py            # every dataset

The notebook is better for a *human* (it shows images). This script is
better for an *agent* (fixed JSON and exit code).  Both use the check
logic in :mod:`radharmony.integrity`, so they stay the same.

Paths come from ``configs/data_paths.json`` (see ``scripts/dataset_paths.py``).
Edit that file to match your NAS layout.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable

from radharmony.dataset import (
    BRAXDataset,
    BRAXPNGDataset,
    ChestXray14TrainDataset,
    ChestXray14TestDataset,
    ChestXray14BboxDataset,
    ChestXDet10TrainDataset,
    ChestXDet10TestDataset,
    CheXphotoTrainDataset,
    CheXphotoValidDataset,
    CheXpertTrainDataset,
    CheXpertValidDataset,
    CheXpertPlusDataset,
    CheXpertPlusTrainDataset,
    CheXpertPlusValidDataset,
    CheXlocalizeDataset,
    CheXlocalizeValDataset,
    VQARadDataset,
    VQARadTrainDataset,
    VQARadTestDataset,
    ChexInstructVQADataset,
    ChexInstructVQATrainDataset,
    ChexInstructVQAValDataset,
    ChexInstructVQATestDataset,
    GEMeXVQADataset,
    ROCODataset,
    ROCOTrainDataset,
    ROCOValidationDataset,
    ROCOTestDataset,
    ROCOv2Dataset,
    ROCOv2TrainDataset,
    ROCOv2ValidationDataset,
    ROCOv2TestDataset,
    MedPMCDataset,
    MIMICCXRDataset,
    MIMICCXRTrainDataset,
    MIMICCXRValidateDataset,
    MIMICCXRTestDataset,
    MIMICCXRJPGDataset,
    MIMICCXRJPGTrainDataset,
    MIMICCXRJPGValidateDataset,
    MIMICCXRJPGTestCheXpertDataset,
    MIMICCXRJPGTestDataset,
    Node21Dataset,
    Node21OriginalDataset,
    OpenICXRDataset,
    PadChestDataset,
    PadChestGRDataset,
    PadChestGRTrainDataset,
    PadChestGRValidationDataset,
    PadChestGRTestDataset,
    ReXGradientTrainDataset,
    ReXGradientValidDataset,
    ReXGradientTestDataset,
    RANZCRClipDataset,
    RANZCRClipTrainDataset,
    RANZCRClipTestDataset,
    VinDrCXRTrainDataset,
    VinDrCXRTestDataset,
    VinDrPCXRDataset,
    VinDrPCXRTrainDataset,
    VinDrPCXRTestDataset,
    ChestImaGenomeGoldDataset,
    ChestImaGenomeSilverDataset,
    ChestImaGenomeSilverTrainDataset,
    ChestImaGenomeSilverValidDataset,
    ChestImaGenomeSilverTestDataset,
    SIIMACRPTXTrainDataset,
    SIIMACRPTXTestDataset,
    SIIMCOVID19TrainDataset,
    SIIMCOVID19TestDataset,
    RSNAPneumoniaDataset,
    RSNAPneumoniaKaggleTrainDataset,
    RSNAPneumoniaKaggleTestDataset,
    MontgomeryCXRDataset,
    ShenzhenCXRDataset,
    QaTaCOV19Dataset,
    QaTaCOV19TrainDataset,
    QaTaCOV19TestDataset,
    MedSegCOVIDCTDataset,
    MedSegCOVIDCTValDataset,
    TAIXRayDataset,
    TAIXRayTrainDataset,
    TAIXRayValDataset,
    TAIXRayTestDataset,
    TAIXRay512Dataset,
    TAIXRay512TrainDataset,
    TAIXRay512ValDataset,
    TAIXRay512TestDataset,
    RSNAPEDetectionTrainDataset,
    RSNAPEDetectionTestDataset,
    RSNABoneAgeTrainDataset,
    RSNABoneAgeValDataset,
    RSNABoneAgeTestDataset,
    RSNA2022CervicalSpineTrainDataset,
    RSNA2022CervicalSpineTestDataset,
    RSNA2022CervicalSpineBboxDataset,
    RSNA2024LumbarSpineTrainDataset,
    RSNA2024LumbarSpineTestDataset,
    RSNAAbdominalTrauma2023Dataset,
    RSNAAbdominalTrauma2023TrainDataset,
    RSNAAbdominalTrauma2023TestDataset,
    CTRATEDataset,
    CTRATETrainDataset,
    CTRATEValidDataset,
    RadChestCTDataset,
    RadChestCTTrainDataset,
    RadChestCTValidDataset,
    RadChestCTTestDataset,
    MIMICExtCXRQBADataset,
    MIMICExtCXRQBATestDataset,
    MIMICExtCXRQBATrainDataset,
    MIMICExtCXRQBAValidateDataset,
    MSCXRDataset,
    MSCXRTDataset,
    MSCXRTestDataset,
    MSCXRTrainDataset,
    MSCXRValDataset,
)
from radharmony.integrity import IntegrityReport, check_dataset, print_report, subsample_dataset

# Paths come from configs/data_paths.json, the one source for scripts and agents. A dataset with no entry
# (a public clone fills only the datasets it checks) gets the MISSING marker, not an error.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset_paths import MISSING  # noqa: E402
from dataset_paths import paths as _paths  # noqa: E402


def P(key, *args, **kwargs):  # noqa: N802  (the short name keeps the spec list readable)
    """Return the path arguments of ``key``, or ``{MISSING: <hint>}`` when it has no entry."""
    return _paths(key, *args, missing_ok=True, **kwargs)


def missing_paths(spec) -> str | None:
    """Return the hint when the spec has no paths, else None."""
    try:
        kwargs = spec.kwargs() if callable(spec.kwargs) else spec.kwargs
    except Exception:  # a "not_on_nas" lambda: run_one reports it
        return None
    return kwargs.get(MISSING)

# Scratch dir for datasets that rasterize/decode masks to PNG on the fly.
MASK_OUT = os.path.expanduser("~/.cache/radharmony/verify_masks")


@dataclass
class DatasetSpec:
    """One dataset to check: a registry key, the class, its constructor
    kwargs, and an optional DataLoader-worker override (3-D volumes use fewer
    workers to bound memory).

    ``kwargs`` can also be a function that returns the kwargs. Use a function
    when ``P(key)`` raises for the entry (for example "not_on_nas"). The run
    then reports ERROR for that dataset only.

    ``default=False`` keeps the spec out of a run with no ``--datasets``. Use
    it when the status of the data in ``configs/data_paths.json`` is
    "not_on_nas" or "restricted" (not every user can read it). Then the
    default run can still pass. Also use it for a very slow dataset, or for a
    spec that fails on a known data problem.
    """

    key: str
    cls: type
    kwargs: dict | Callable[[], dict] = field(default_factory=dict)
    workers: int | None = None
    default: bool = True


#: Old check names (before 2026-11-04) and their registry keys. They still work, with a warning.
OLD_NAMES = {
    "vindr_train": "vindr_cxr_train",
    "vindr_test": "vindr_cxr_test",
    "ranzcr": "ranzcr_clip",
    "ranzcr_mask": "ranzcr_clip_mask",
    "montgomery": "montgomery_cxr",
    "shenzhen": "shenzhen_cxr",
    "rsna_pe": "rsna_pe_detection_train",
    "rsna_pe_test": "rsna_pe_detection_test",
    "rsna_cervical": "rsna_2022_cervical_spine_train",
    "rsna_cervical_test": "rsna_2022_cervical_spine_test",
    "rsna_cervical_bbox": "rsna_2022_cervical_spine_bbox",
    "rsna_lumbar": "rsna_2024_lumbar_spine_train",
    "rsna_lumbar_test": "rsna_2024_lumbar_spine_test",
    "rsna_abdominal": "rsna_abdominal_trauma_2023",
    "rsna_abdominal_train": "rsna_abdominal_trauma_2023_train",
    "rsna_abdominal_test": "rsna_abdominal_trauma_2023_test",
}


def resolve_names(names: list[str]) -> list[str]:
    """Return the names with each old check name changed to its registry key, and warn for each one."""
    out = []
    for name in names:
        if name in OLD_NAMES:
            print(f"warning: {name} is an old name. Use {OLD_NAMES[name]}.", file=sys.stderr)
        out.append(OLD_NAMES.get(name, name))
    return out


def get_specs() -> list[DatasetSpec]:
    """The public (external) dataset registry.  Mirrors the notebook's
    ``build(...)`` calls one-to-one."""
    S = DatasetSpec
    return [
        # ── VQA ──
        S("vqa_rad_train", VQARadTrainDataset, dict(
            **P("vqa_rad_train"), output_question_type=True)),
        S("vqa_rad_test", VQARadTestDataset, dict(
            **P("vqa_rad_test"), output_question_type=True)),
        S("vqa_rad", VQARadDataset, dict(
            **P("vqa_rad"), output_question_type=True)),
        S("chexinstruct_vqa_test", ChexInstructVQATestDataset, dict(
            **P("chexinstruct_vqa_test"), output_struct=True, output_question_type=True)),
        S("chexinstruct_vqa_val", ChexInstructVQAValDataset, dict(
            **P("chexinstruct_vqa_val"), output_struct=True, output_question_type=True)),
        S("chexinstruct_vqa_train", ChexInstructVQATrainDataset, dict(
            **P("chexinstruct_vqa_train"), output_struct=True, output_question_type=True)),
        S("chexinstruct_vqa", ChexInstructVQADataset, dict(
            **P("chexinstruct_vqa"), output_struct=True, output_question_type=True)),
        S("gemex_vqa", GEMeXVQADataset, dict(
            **P("gemex_vqa"), output_struct=True, output_question_type=True)),
        # Captioning. The ROCO (v1) data is not on the NAS, so P() raises at run time.
        S("roco", ROCODataset, lambda: dict(
            **P("roco"), output_struct=True, output_question_type=True), default=False),
        S("roco_train", ROCOTrainDataset, lambda: dict(**P("roco_train")), default=False),
        S("roco_validation", ROCOValidationDataset, lambda: dict(**P("roco_validation")), default=False),
        S("roco_test", ROCOTestDataset, lambda: dict(**P("roco_test")), default=False),
        # ROCOv2: only the test images are extracted from the parquet files (2026-11-02). The train and
        # validation images need scripts/materialize_rocov2.py with write access to the folder.
        S("rocov2", ROCOv2Dataset, dict(**P("rocov2")), default=False),
        S("rocov2_train", ROCOv2TrainDataset, dict(**P("rocov2_train")), default=False),
        S("rocov2_validation", ROCOv2ValidationDataset, dict(**P("rocov2_validation")), default=False),
        S("rocov2_test", ROCOv2TestDataset, dict(**P("rocov2_test"))),
        # Only the "high-precision radiology" subset (the app filter) has image files on
        # disk (scripts/materialize_medpmc_subset.py). Most of the 11M rows have no image file.
        S("medpmc", MedPMCDataset, dict(
            **P("medpmc"), subsets=["singlepanel", "subfigure"], radiology_only=True,
            require_tagger_agreement=True, min_clip_prob=0.9)),
        # ── 2-D CXR ──
        S("chexpert_train", CheXpertTrainDataset, dict(
            **P("chexpert_train"), output_cls=True, drop_uncertain=False)),
        S("chexpert_valid", CheXpertValidDataset, dict(
            **P("chexpert_valid"), output_cls=True)),
        S("chexpert_plus_train", CheXpertPlusTrainDataset, dict(
            **P("chexpert_plus_train"), output_cls=True,
            output_report=True, drop_uncertain=False)),
        S("chexpert_plus_valid", CheXpertPlusValidDataset, dict(
            **P("chexpert_plus_valid"), output_cls=True,
            output_report=True, drop_uncertain=False)),
        S("chexpert_plus", CheXpertPlusDataset, dict(
            **P("chexpert_plus"), output_cls=True,
            output_report=True, drop_uncertain=False)),
        S("chexphoto_train", CheXphotoTrainDataset, dict(
            **P("chexphoto_train"), output_cls=True)),
        S("chexphoto_valid", CheXphotoValidDataset, dict(
            **P("chexphoto_valid"), output_cls=True)),
        S("chexlocalize", CheXlocalizeDataset, dict(
            **P("chexlocalize", only=("base_image_dir", "csv_path")), output_cls=True)),
        S("chexlocalize_val", CheXlocalizeValDataset, dict(
            **P("chexlocalize_val", only=("base_image_dir", "csv_path")), output_cls=True)),
        S("chexlocalize_mask", CheXlocalizeDataset, dict(
            **P("chexlocalize"),
            mask_output_dir=os.path.join(MASK_OUT, "chexlocalize"),
            output_cls=True, output_mask=True)),
        S("mimic_cxr", MIMICCXRDataset, dict(
            **P("mimic_cxr"), output_cls=True, output_report=True, drop_uncertain=False)),
        S("mimic_cxr_train", MIMICCXRTrainDataset, dict(
            **P("mimic_cxr_train"), output_cls=True, output_report=True, drop_uncertain=False)),
        S("mimic_cxr_validate", MIMICCXRValidateDataset, dict(
            **P("mimic_cxr_validate"), output_cls=True, output_report=True, drop_uncertain=False)),
        S("mimic_cxr_test", MIMICCXRTestDataset, dict(
            **P("mimic_cxr_test"), output_cls=True, output_report=True, drop_uncertain=False)),
        S("mimic_cxr_jpg", MIMICCXRJPGDataset, dict(
            **P("mimic_cxr_jpg"), output_cls=True)),
        S("mimic_cxr_jpg_train", MIMICCXRJPGTrainDataset, dict(
            **P("mimic_cxr_jpg_train"), output_cls=True)),
        S("mimic_cxr_jpg_validate", MIMICCXRJPGValidateDataset, dict(
            **P("mimic_cxr_jpg_validate"), output_cls=True)),
        S("mimic_cxr_jpg_test", MIMICCXRJPGTestDataset, dict(
            **P("mimic_cxr_jpg_test"), output_cls=True)),
        S("mimic_cxr_jpg_test_chexpert", MIMICCXRJPGTestCheXpertDataset, dict(
            **P("mimic_cxr_jpg_test_chexpert"), output_cls=True)),
        # MS-CXR, MS-CXR-T and MIMIC-Ext-CXR-QBA: public classes on MIMIC-CXR images (D-39).
        S("ms_cxr_train", MSCXRTrainDataset, dict(**P("ms_cxr_train"), output_cls=True, output_bbox=True)),
        S("ms_cxr_val", MSCXRValDataset, dict(**P("ms_cxr_val"), output_cls=True, output_bbox=True)),
        S("ms_cxr_test", MSCXRTestDataset, dict(**P("ms_cxr_test"), output_cls=True, output_bbox=True)),
        S("ms_cxr", MSCXRDataset, dict(**P("ms_cxr"), output_cls=True, output_bbox=True)),
        S("ms_cxr_t", MSCXRTDataset, dict(**P("ms_cxr_t"), output_previous=True)),
        # VQA: the check reads the image. max_studies caps the slow qa.zip step.
        S("mimic_ext_cxr_qba", MIMICExtCXRQBADataset, dict(
            **P("mimic_ext_cxr_qba"), image_ext=".dcm", max_studies=20,
            output_struct=True, output_question_type=True)),
        S("mimic_ext_cxr_qba_train", MIMICExtCXRQBATrainDataset, dict(
            **P("mimic_ext_cxr_qba_train"), image_ext=".dcm", max_studies=20,
            output_struct=True, output_question_type=True)),
        S("mimic_ext_cxr_qba_validate", MIMICExtCXRQBAValidateDataset, dict(
            **P("mimic_ext_cxr_qba_validate"), image_ext=".dcm", max_studies=20,
            output_struct=True, output_question_type=True)),
        S("mimic_ext_cxr_qba_test", MIMICExtCXRQBATestDataset, dict(
            **P("mimic_ext_cxr_qba_test"), image_ext=".dcm", max_studies=20,
            output_struct=True, output_question_type=True)),
        S("chestxray14_train", ChestXray14TrainDataset, dict(
            **P("chestxray14_train"), output_cls=True)),
        S("chestxray14_test", ChestXray14TestDataset, dict(
            **P("chestxray14_test"), output_cls=True)),
        S("chestxray14_bbox", ChestXray14BboxDataset, dict(
            **P("chestxray14_bbox"), output_cls=True, output_bbox=True)),
        S("chestxdet10_train", ChestXDet10TrainDataset, dict(
            **P("chestxdet10_train"), output_cls=True, output_bbox=True)),
        S("chestxdet10_test", ChestXDet10TestDataset, dict(
            **P("chestxdet10_test"), output_cls=True, output_bbox=True)),
        S("node21", Node21Dataset, dict(
            **P("node21"), output_cls=True, output_bbox=True)),
        S("node21_original", Node21OriginalDataset, dict(
            **P("node21_original"), output_cls=True, output_bbox=True)),
        S("padchest", PadChestDataset, dict(
            **P("padchest"), output_cls=True, output_report=True)),
        S("padchest_gr_train", PadChestGRTrainDataset, dict(
            **P("padchest_gr_train"), output_cls=True, output_report=True, output_bbox=True), default=False),
        S("padchest_gr_validation", PadChestGRValidationDataset, dict(
            **P("padchest_gr_validation"), output_cls=True, output_report=True,
            output_bbox=True), default=False),
        S("padchest_gr_test", PadChestGRTestDataset, dict(
            **P("padchest_gr_test"), output_cls=True, output_report=True, output_bbox=True), default=False),
        S("padchest_gr", PadChestGRDataset, dict(
            **P("padchest_gr"), output_cls=True, output_report=True, output_bbox=True), default=False),
        S("brax", BRAXDataset, dict(
            **P("brax"), uncertain_strategy="u_zeros", output_cls=True)),
        S("brax_png", BRAXPNGDataset, dict(
            **P("brax_png"), uncertain_strategy="u_zeros", output_cls=True)),
        S("openi_cxr", OpenICXRDataset, dict(
            **P("openi_cxr"), output_cls=True, output_report=True)),
        S("rexgradient_valid", ReXGradientValidDataset, dict(
            **P("rexgradient_valid"), output_report=True)),
        S("rexgradient_test", ReXGradientTestDataset, dict(
            **P("rexgradient_test"), output_report=True)),
        S("rexgradient_train", ReXGradientTrainDataset, dict(
            **P("rexgradient_train"), output_report=True)),
        S("vindr_cxr_train", VinDrCXRTrainDataset, dict(
            **P("vindr_cxr_train"), output_cls=True, output_bbox=True)),
        S("vindr_cxr_test", VinDrCXRTestDataset, dict(
            **P("vindr_cxr_test"), output_cls=True, output_bbox=True)),
        S("vindr_pcxr", VinDrPCXRDataset, dict(
            **P("vindr_pcxr"), output_cls=True, output_bbox=True)),
        S("vindr_pcxr_train", VinDrPCXRTrainDataset, dict(
            **P("vindr_pcxr_train"), output_cls=True, output_bbox=True)),
        S("vindr_pcxr_test", VinDrPCXRTestDataset, dict(
            **P("vindr_pcxr_test"), output_cls=True, output_bbox=True)),
        S("chest_imagenome_gold", ChestImaGenomeGoldDataset, dict(
            **P("chest_imagenome_gold"), output_bbox=True)),
        # "valid" is the smallest official split of the slow silver release.
        S("chest_imagenome_silver_valid", ChestImaGenomeSilverValidDataset, dict(
            **P("chest_imagenome_silver_valid"), output_bbox=True)),
        # Slow: harmonize() reads one DICOM header per image. Run with --datasets.
        S("chest_imagenome_silver_train", ChestImaGenomeSilverTrainDataset, dict(
            **P("chest_imagenome_silver_train"), output_bbox=True), default=False),
        S("chest_imagenome_silver_test", ChestImaGenomeSilverTestDataset, dict(
            **P("chest_imagenome_silver_test"), output_bbox=True), default=False),
        S("chest_imagenome_silver", ChestImaGenomeSilverDataset, dict(
            **P("chest_imagenome_silver"), output_bbox=True), default=False),
        S("siim_acr_ptx", SIIMACRPTXTrainDataset, dict(
            **P("siim_acr_ptx_train"),
            mask_output_dir=os.path.join(MASK_OUT, "siim_acr"),
            output_cls=True, output_mask=True)),
        # siim_acr_ptx_test, siim_covid19_test and rsna_pneumonia_kaggle_test have no labels
        # (D-25), so their specs check only the image.
        S("siim_acr_ptx_test", SIIMACRPTXTestDataset, dict(**P("siim_acr_ptx_test"))),
        S("siim_covid19", SIIMCOVID19TrainDataset, dict(
            **P("siim_covid19_train"), output_cls=True, output_bbox=True)),
        S("siim_covid19_test", SIIMCOVID19TestDataset, dict(**P("siim_covid19_test"))),
        S("rsna_pneumonia", RSNAPneumoniaKaggleTrainDataset, dict(
            **P("rsna_pneumonia_kaggle_train"), output_cls=True, output_bbox=True)),
        S("rsna_pneumonia_kaggle_test", RSNAPneumoniaKaggleTestDataset, dict(**P("rsna_pneumonia_kaggle_test"))),
        # This spec checks the registry key rsna_pneumonia (the adjudicated JSON). The spec
        # rsna_pneumonia above checks the Kaggle train class. The JSON data is not on the NAS.
        S("rsna_pneumonia_adjudicated", RSNAPneumoniaDataset, lambda: dict(
            **P("rsna_pneumonia"), output_cls=True, output_bbox=True), default=False),
        S("ranzcr_clip_train", RANZCRClipTrainDataset, dict(
            **P("ranzcr_clip_train", only=("base_image_dir", "csv_path")), output_cls=True)),
        S("ranzcr_clip_test", RANZCRClipTestDataset, dict(**P("ranzcr_clip_test"))),
        S("ranzcr_clip", RANZCRClipDataset, dict(
            **P("ranzcr_clip", only=("base_image_dir", "csv_path")),
            include_test_split=False, output_cls=True)),
        S("ranzcr_clip_mask", RANZCRClipDataset, dict(
            **P("ranzcr_clip"),
            mask_output_dir=os.path.join(MASK_OUT, "ranzcr"),
            include_test_split=False, output_cls=True, output_mask=True)),
        S("taix_ray_512_train", TAIXRay512TrainDataset, dict(
            **P("taix_ray_512_train"), label_mode="ordinal", output_cls=True)),
        S("taix_ray_512_val", TAIXRay512ValDataset, dict(
            **P("taix_ray_512_val"), label_mode="ordinal", output_cls=True)),
        S("taix_ray_512_test", TAIXRay512TestDataset, dict(
            **P("taix_ray_512_test"), label_mode="ordinal", output_cls=True)),
        S("taix_ray_train", TAIXRayTrainDataset, dict(
            **P("taix_ray_train"), label_mode="ordinal", output_cls=True)),
        S("taix_ray_val", TAIXRayValDataset, dict(
            **P("taix_ray_val"), label_mode="ordinal", output_cls=True)),
        S("taix_ray_test", TAIXRayTestDataset, dict(
            **P("taix_ray_test"), label_mode="ordinal", output_cls=True)),
        S("taix_ray", TAIXRayDataset, dict(
            **P("taix_ray"), label_mode="ordinal", output_cls=True)),
        S("taix_ray_512", TAIXRay512Dataset, dict(
            **P("taix_ray_512"), label_mode="ordinal", output_cls=True)),
        S("montgomery_cxr", MontgomeryCXRDataset, dict(
            **P("montgomery_cxr"),
            mask_output_dir=os.path.join(MASK_OUT, "montgomery"),
            output_cls=True, output_mask=True, output_report=True)),
        S("shenzhen_cxr", ShenzhenCXRDataset, dict(
            **P("shenzhen_cxr"),
            mask_output_dir=os.path.join(MASK_OUT, "shenzhen"),
            output_cls=True, output_mask=True, output_report=True)),
        # One class per split. No mask_output_dir: QaTa ships ready-made
        # binary mask PNGs. The old class QaTaCOV19Dataset
        # (key qata_cov19) reads the folder that you give.
        S("qata_cov19", QaTaCOV19Dataset, dict(
            **P("qata_cov19", "train"), output_cls=True, output_mask=True)),
        S("qata_cov19_train", QaTaCOV19TrainDataset, dict(
            **P("qata_cov19_train"), output_cls=True, output_mask=True)),
        S("qata_cov19_test", QaTaCOV19TestDataset, dict(
            **P("qata_cov19_test"), output_cls=True, output_mask=True)),
        # CT slices. The lesion flags are real multi-label (not a constant like QaTa).
        S("medseg_covid_ct_d1", MedSegCOVIDCTDataset, dict(
            **P("medseg_covid_ct", "dataset1"),
            subset="dataset1", output_cls=True, output_mask=True)),
        S("medseg_covid_ct_d2", MedSegCOVIDCTDataset, dict(
            **P("medseg_covid_ct", "dataset2"),
            subset="dataset2", output_cls=True, output_mask=True)),
        # The slices of the val volume are not made yet, and they need write access to the folder (2026-11-02).
        S("medseg_covid_ct_val", MedSegCOVIDCTValDataset, dict(**P("medseg_covid_ct_val")), default=False),
        S("rsna_bone_age", RSNABoneAgeTrainDataset, dict(
            **P("rsna_bone_age_train"), output_reg=True)),
        S("rsna_bone_age_val", RSNABoneAgeValDataset, dict(
            **P("rsna_bone_age_val"), output_reg=True)),
        S("rsna_bone_age_test", RSNABoneAgeTestDataset, lambda: dict(**P("rsna_bone_age_test")), default=False),
        # ── 3-D (fewer workers to bound memory) ──
        S("rsna_pe_detection_train", RSNAPEDetectionTrainDataset, dict(
            **P("rsna_pe_detection_train"), output_cls=True), workers=2),
        S("rsna_pe_detection_test", RSNAPEDetectionTestDataset, dict(**P("rsna_pe_detection_test")), workers=2),
        S("rsna_2022_cervical_spine_train", RSNA2022CervicalSpineTrainDataset, dict(
            **P("rsna_2022_cervical_spine_train"),
            output_cls=True, output_mask=True), workers=2),
        # The 3 studies in the Kaggle test.csv have no image folder (V-A-29), so this
        # spec fails. Run it with --datasets.
        S("rsna_2022_cervical_spine_test", RSNA2022CervicalSpineTestDataset, dict(
            **P("rsna_2022_cervical_spine_test")), workers=2, default=False),
        S("rsna_2022_cervical_spine_bbox", RSNA2022CervicalSpineBboxDataset, dict(
            **P("rsna_2022_cervical_spine_bbox"),
            output_cls=True, output_bbox=True), workers=2),
        S("rsna_2024_lumbar_spine_train", RSNA2024LumbarSpineTrainDataset, dict(
            **P("rsna_2024_lumbar_spine_train"),
            output_cls=True, output_bbox=True), workers=2),
        S("rsna_2024_lumbar_spine_test", RSNA2024LumbarSpineTestDataset, dict(
            **P("rsna_2024_lumbar_spine_test")), workers=2),
        S("rsna_abdominal_trauma_2023", RSNAAbdominalTrauma2023Dataset, dict(
            **P("rsna_abdominal_trauma_2023"), output_cls=True), workers=2),
        S("rsna_abdominal_trauma_2023_train", RSNAAbdominalTrauma2023TrainDataset, dict(
            **P("rsna_abdominal_trauma_2023_train"), output_cls=True), workers=2),
        # The Kaggle test_images folder has 3 of the 6 test series, with 1 slice
        # each, so this spec fails. Run it with --datasets.
        S("rsna_abdominal_trauma_2023_test", RSNAAbdominalTrauma2023TestDataset, dict(
            **P("rsna_abdominal_trauma_2023_test")), workers=2, default=False),
        # The folder picks the split: not for benchmarks (D-09).
        S("ct_rate", CTRATEDataset, dict(**P("ct_rate"), output_cls=True), workers=2),
        S("ct_rate_train", CTRATETrainDataset, dict(
            **P("ct_rate_train"), output_cls=True), workers=2),
        S("ct_rate_valid", CTRATEValidDataset, dict(
            **P("ct_rate_valid"), output_cls=True), workers=2),
        S("radchestct_train", RadChestCTTrainDataset, dict(
            **P("radchestct_train"), output_cls=True, output_bbox=True), workers=2),
        S("radchestct_valid", RadChestCTValidDataset, dict(
            **P("radchestct_valid"), output_cls=True, output_bbox=True), workers=2),
        S("radchestct_test", RadChestCTTestDataset, dict(
            **P("radchestct_test"), output_cls=True, output_bbox=True), workers=2),
        # The label file picks the split: not for benchmarks (D-09).
        S("radchestct", RadChestCTDataset, dict(
            **P("radchestct"), output_cls=True, output_bbox=True), workers=2),
    ]


def run_one(spec: DatasetSpec, max_samples: int | None, default_workers: int):
    """Instantiate -> sub-sample -> integrity-check a single dataset."""
    print("\n" + "#" * 72)
    print(f"# {spec.key}  ({spec.cls.__name__})")
    print("#" * 72)
    try:
        kwargs = spec.kwargs() if callable(spec.kwargs) else spec.kwargs
        ds = spec.cls(cache_dir=None, **kwargs)
    except Exception as e:
        traceback.print_exc()
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
    try:
        return check_dataset(ds, num_workers=workers)
    except Exception as e:  # one odd dataset must not stop the other datasets
        traceback.print_exc()
        return IntegrityReport(
            dataset=spec.cls.__name__, status="ERROR",
            error_message=f"check_dataset failed: {type(e).__name__}: {e}")


def print_summary(rows: list[tuple[str, "IntegrityReport"]]) -> None:
    """Final aligned status table across all datasets."""
    icon = {"OK": "✅", "WARN": "⚠️", "ERROR": "❌"}
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"{'key':22s} {'status':7s} {'ok/total':>14s} {'img_range':>18s}  time")
    print("-" * 78)
    for key, r in rows:
        rng = f"[{r.img_min:.2f},{r.img_max:.2f}]" if r.has_range else "-"
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
    if args.max_samples < 0:
        p.error("--max-samples must be 0 or more (0 = every sample)")

    if args.list:
        for s in specs:
            note = "" if s.default else "  (only with --datasets)"
            print(f"  {s.key:22s} {s.cls.__name__}{note}")
        return 0

    def keys_of(arg: str) -> list[str]:
        keys = resolve_names([k.strip() for k in arg.split(",") if k.strip()])
        if not keys:
            p.error("no dataset key given\nrun with --list to see valid keys")
        unknown = [k for k in keys if k not in by_key]
        if unknown:
            p.error(f"unknown dataset key(s): {', '.join(unknown)}\n"
                    f"run with --list to see valid keys")
        return keys

    selected = [s for s in specs if s.default]
    if args.datasets:
        selected = [by_key[k] for k in keys_of(args.datasets)]
        hints = [missing_paths(s) for s in selected if missing_paths(s)]
        if hints:
            p.error(hints[0])
    else:
        no_path = [s for s in selected if missing_paths(s)]
        if no_path:
            print(f"skipped {len(no_path)} dataset(s) with no path in the data paths file "
                  f"({', '.join(s.key for s in no_path[:5])}{', ...' if len(no_path) > 5 else ''}).")
            selected = [s for s in selected if not missing_paths(s)]
    if args.skip:
        drop = set(keys_of(args.skip))
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
            # The same order as cls (D-05), so each count gets the right name.
            label_cols = sorted(getattr(spec.cls, "LABEL_COLS", []) or [])
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
            print(payload)  # Real stdout, pure JSON.
        else:
            with open(args.json, "w") as f:
                f.write(payload)
            print(f"\nWrote JSON report -> {args.json}")

    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
