"""Harmonizer for MR-RATE: brain and spine MRI volumes with radiology reports.

Source: https://huggingface.co/datasets/Forithmus/MR-RATE (gated, request
access). License: CC BY-NC-SA-4.0. Built by Forithmus and the University of
Zurich, with NVIDIA and Istanbul Medipol.

705,254 MRI volumes from 98,334 studies of 83,425 patients. Each study has a
structured report. The design follows CT-RATE: one row for each series. The
study reports and labels repeat across the series of a study.

Layout. Each study is one zip in the HF repo. After extraction::

    <base_dir>/
      meta/
        splits.csv                                  # patient_uid, study_uid, batch_id, split
        metadata/batch{00..27}_metadata.csv         # 316 cols, one row per series
        reports/batch{00..27}_reports.csv           # study_uid + 5 report sections
        pathology_labels/mrrate_labels.csv          # 37 binary pathology columns
      zips/mri/<batch_id>/<study_uid>.zip           # as downloaded
      mri/<batch_id>/<study_uid>/                   # after materialize_mrrate.py
        img/<study_uid>_<series_id>.nii.gz
        seg/<study_uid>_<series_id>_brain-mask.nii.gz
        seg/<study_uid>_<series_id>_defacing-mask.nii.gz

``series_id`` is the file name stem (``t1w-raw-sag``, ...). So image and mask
paths are a direct join, with no glob over 705K files.

The volumes are 3-D NIfTI, not 2-D images. Treat them like the RSNA
lumbar-spine data, not like CXR.

Three facts measured on the metadata on 2026-08-22:

* ``is_derived`` and ``is_localizer`` are False in all 705,254 rows. A filter
  on them does nothing. They stay as output columns.
* The nvseg brain parcellation (132 regions) exists only for the center
  series. Every series has a whole-body (``-wb``) segmentation. With
  ``nvseg-brain`` and without ``center_only=True``, most rows have no mask
  file, and ``require_images=True`` drops them.
* ``classified_modality`` is mostly T1w, T2w, FLAIR and SWI. MRA has only 164
  series, so do not plan an angiography experiment.

97,896 of 98,334 studies have pathology labels. The other 438 have NaN in all
37 label columns, not 0.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from ..base import BaseHarmonizer

#: The 37 pathology categories, each grounded to SNOMED CT or RadLex upstream.
#: Order matches mrrate_labels.csv so a positional read stays aligned.
PATHOLOGY_LABELS = [
    "Cerebral infarction", "Cerebral hemorrhage",
    "Metastatic malignant neoplasm to brain", "Intracranial meningioma",
    "Demyelinating disease of central nervous system",
    "Herniation of nucleus pulposus", "Spinal cord compression", "Lacunar infarct",
    "Silent micro-hemorrhage of brain", "Cavernous hemangioma",
    "Subdural intracranial hemorrhage", "Gliosis", "Cerebral atrophy",
    "Encephalomalacia", "Arachnoid cyst", "Empty sella syndrome",
    "Intracranial aneurysm", "Chiari malformation", "Schwannoma",
    "Cyst of pineal gland", "Hemangioma of vertebral column", "Rathke's pouch cyst",
    "Cerebral edema", "Spinal stenosis", "Mastoiditis", "Chronic mastoiditis",
    "Ventriculomegaly", "Cerebellar degeneration", "Mega cisterna magna",
    "Structure of cave of septum pellucidum", "Hyperostosis of skull",
    "Watershed infarct", "Choroid plexus cyst", "Foraminal Spinal Stenosis",
    "Lipoma of brain", "Glioma", "Pituitary adenoma",
]

MODALITIES = ("T1w", "T2w", "FLAIR", "SWI", "MRA")
SPLITS = ("train", "val", "test")

#: Series-level metadata worth carrying without reopening the 1.2 GB CSVs.
_META_COLS = ["patient_uid", "study_uid", "series_id", "classified_modality",
              "acquisition_plane", "sequence_family", "is_center_modality",
              "is_derived", "is_localizer", "SeriesNumber", "FieldStrength_T",
              "Manufacturer", "Patient'sAge", "Patient'sSex"]

#: Source column -> output column of the series metadata that the table keeps.
_EXTRA_SOURCE_COLS = {"split": "split", "batch_id": "batch_id",
                      "classified_modality": "classified_modality",
                      "acquisition_plane": "acquisition_plane",
                      "sequence_family": "sequence_family",
                      "is_center_modality": "is_center_modality",
                      "is_derived": "is_derived", "is_localizer": "is_localizer",
                      "FieldStrength_T": "field_strength_t", "Manufacturer": "manufacturer",
                      "Patient'sAge": "patient_age", "Patient'sSex": "patient_sex"}


class MRRATEHarmonizer(BaseHarmonizer):
    """Harmonize MR-RATE into the RadHarmony schema, one row per series.

    Args:
        base_dir: Dataset root containing ``meta/`` and (once materialised) ``mri/``.
        splits: Subset of ``("train", "val", "test")``. ``None`` keeps all.
        modalities: Subset of :data:`MODALITIES`. ``None`` keeps all.
        center_only: Keep only the center (T1w) series of each study. This gives
            one row for each study (98,334 rows).
        require_images: Drop rows whose ``.nii.gz`` is not on disk. The default
            is False, so the table works from metadata alone.
        with_masks: Populate ``mask_path``.
        mask_source: Which mask ``mask_path`` points at.
            ``"brain"``      HD-BET brain mask (binary, from the main repo).
            ``"deface"``     Quickshear defacing mask (binary, from the main repo).
            ``"nvseg-brain"`` NV-Segment-CTMR brain parcellation, 132 regions,
                             center series only, from the MR-RATE-nvseg-ctmr repo.
            ``"nvseg-wb"``   NV-Segment-CTMR whole-body segmentation, all series
        report_section: Text for ``report``: ``"full"``, ``"findings"`` or
            ``"impression"``.
    """

    LABEL_COLS = PATHOLOGY_LABELS
    SERIES_ID_SOURCE_COL = "series_id"
    EXTRA_OUTPUT_COLS = ["split", "batch_id", "classified_modality", "acquisition_plane",
                         "sequence_family", "is_center_modality", "is_derived",
                         "is_localizer", "field_strength_t", "manufacturer",
                         "patient_age", "patient_sex", "impression"]
    #: Series metadata ``Patient'sSex``, renamed to ``patient_sex``. The base step makes
    #: ``sex`` (D-26). No ``age_years``: nobody checked the unit of ``Patient'sAge``.
    SEX_SOURCE_COL = "patient_sex"

    def __init__(
        self,
        base_dir: str = None,
        base_image_dir: str = None,
        splits: list[str] | None = None,
        modalities: list[str] | None = None,
        center_only: bool = False,
        require_images: bool = False,
        with_masks: bool = True,
        mask_source: str = "brain",
        report_section: str = "full",
    ):
        super().__init__(csv_path=None)
        self.base_dir = os.path.expanduser(base_dir) if base_dir else None
        self.base_image_dir = base_image_dir
        self.splits = list(splits) if splits else None
        self.modalities = list(modalities) if modalities else None
        self.center_only = center_only
        self.require_images = require_images
        self.with_masks = with_masks
        if mask_source not in ("brain", "deface", "nvseg-brain", "nvseg-wb"):
            raise ValueError("mask_source must be brain, deface, nvseg-brain or nvseg-wb")
        self.mask_source = mask_source
        if report_section not in ("full", "findings", "impression"):
            raise ValueError("report_section must be full, findings or impression")
        self.report_section = report_section

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap.pop("csv_path", None)  # not an argument of this class (V-H2-1)
        snap.update({n: getattr(self, n) for n in (
            "base_dir", "splits", "modalities", "center_only", "require_images",
            "with_masks", "mask_source", "report_section")})
        return snap

    # ------------------------------------------------------------------ loading
    def _meta_dir(self) -> str:
        return os.path.join(self.base_dir, "meta")

    def _read_series(self) -> pd.DataFrame:
        """Read the series metadata. Use the cached parquet if it exists, because the CSVs are 1.2 GB."""
        cache = os.path.join(self._meta_dir(), "series_index.parquet")
        if os.path.isfile(cache):
            d = pd.read_parquet(cache)
            missing = [c for c in ("patient_uid", "study_uid", "series_id",
                                   "classified_modality") if c not in d.columns]
            if not missing:
                return d
        files = sorted(glob.glob(os.path.join(self._meta_dir(), "metadata", "*_metadata.csv")))
        if not files:
            raise FileNotFoundError(
                f"no metadata CSVs under {self._meta_dir()}/metadata. Download them "
                f"first: hf download Forithmus/MR-RATE --repo-type dataset "
                f"--include '*.csv' --local-dir <base_dir>/meta")
        return pd.concat(
            [pd.read_csv(f, usecols=lambda c: c in _META_COLS, low_memory=False)
             for f in files], ignore_index=True)

    def _read_reports(self) -> pd.DataFrame:
        files = sorted(glob.glob(os.path.join(self._meta_dir(), "reports", "*_reports.csv")))
        if not files:
            return pd.DataFrame(columns=["study_uid", "report", "findings", "impression"])
        cols = ["study_uid", "report", "findings", "impression"]
        return pd.concat([pd.read_csv(f, usecols=lambda c: c in cols) for f in files],
                         ignore_index=True)

    def _read_labels(self) -> pd.DataFrame:
        p = os.path.join(self._meta_dir(), "pathology_labels", "mrrate_labels.csv")
        return pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame(columns=["study_uid"])

    def _read_splits(self) -> pd.DataFrame:
        p = os.path.join(self._meta_dir(), "splits.csv")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"missing {p}")
        return pd.read_csv(p)

    # ------------------------------------------------------------------ harmonize steps
    def _read_source(self) -> pd.DataFrame:
        """Read the series metadata and the splits. Keep the series of the chosen splits and modalities.

        This replaces the base csv_path step. MR-RATE has 28 metadata CSVs and
        28 report CSVs, not one table.
        """
        if self.base_dir is None:
            raise ValueError("MRRATEHarmonizer requires base_dir")

        d = self._read_series()
        sp = self._read_splits()
        # batch_id lives only in splits.csv but is part of every image path.
        d = d.merge(sp[["study_uid", "batch_id", "split"]], on="study_uid", how="left")

        if self.splits:
            d = d[d["split"].isin(self.splits)]
        if self.modalities:
            d = d[d["classified_modality"].isin(self.modalities)]
        if self.center_only:
            d = d[d["is_center_modality"].fillna(False).astype(bool)]
        # No early return for 0 series: the later steps give the harmonized columns (V-H2-30).
        # Keep only the columns that the steps use, with their output names.
        cols = ["patient_uid", "study_uid", "series_id"] + list(_EXTRA_SOURCE_COLS)
        d = d[[c for c in cols if c in d.columns]].rename(columns=_EXTRA_SOURCE_COLS)
        return d.reset_index(drop=True)

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["patient_uid"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["study_uid"].astype(str)

    def _stem_and_dir(self) -> tuple[pd.Series, pd.Series]:
        """Return the file stem ``<study_uid>_<series_id>`` and the study folder ``<batch_id>/<study_uid>``."""
        stem = self.df["study_id"] + "_" + self.df["series_id"]
        return stem, self.df["batch_id"].astype(str) + "/" + self.df["study_id"]

    def _build_image_path(self) -> None:
        stem, study_dir = self._stem_and_dir()
        self.df["image_path"] = "mri/" + study_dir + "/img/" + stem + ".nii.gz"

    def _build_mask_path(self) -> None:
        """Make mask_path (with_masks). With require_images, drop the rows whose volume or mask is not on disk.

        The drop runs here, after the label merge, so a dropped study with no label still makes the
        label columns float, as before D-30.
        """
        if self.with_masks:
            self._make_mask_path()
        if self.require_images:
            root = self.base_image_dir or self.base_dir
            # astype(bool): on 0 rows, map() keeps the str dtype, and df[keep] then picks columns (V-H2-30).
            keep = self.df["image_path"].map(lambda p: os.path.isfile(os.path.join(root, p))).astype(bool)
            if self.with_masks:
                # A present volume with an absent mask still crashes the pipeline.
                keep &= self.df["mask_path"].map(os.path.isfile).astype(bool)
            self.df = self.df[keep].reset_index(drop=True)

    def _make_mask_path(self) -> None:
        # The mask path is absolute on purpose. get_data_dict() adds base_path
        # to the image column only. A relative mask_path would reach MONAI's
        # LoadImaged unresolved and the sample would fail.
        root = (self.base_image_dir or self.base_dir).rstrip("/")
        stem, study_dir = self._stem_and_dir()
        if self.mask_source in ("brain", "deface"):
            suffix = {"brain": "_brain-mask", "deface": "_defacing-mask"}[self.mask_source]
            self.df["mask_path"] = root + "/mri/" + study_dir + "/seg/" + stem + suffix + ".nii.gz"
        else:
            # nvseg is a separate repo with its own tree (mri_nvseg/, not mri/).
            # The brain parcellation exists only for the center series.
            kind = "brain" if self.mask_source == "nvseg-brain" else "wb"
            self.df["mask_path"] = (root + "/mri_nvseg/" + study_dir + "/seg/" + stem
                                    + f"_nvseg-ctmr-{kind}.nii.gz")

    def preprocess_masks(self, *args, **kwargs) -> None:
        """Raise ``ValueError``. ``mask_path`` holds NIfTI mask files, not encoded masks to decode."""
        raise ValueError("MR-RATE masks are NIfTI files already. Leave mask_output_dir as None.")

    def preprocess(self, output_dir: str, preprocessor, base_image_dir: str = None, **kwargs) -> None:
        """Run :meth:`BaseHarmonizer.preprocess`. ``base_image_dir`` defaults to the image root.

        The image root is ``self.base_image_dir``, else ``self.base_dir``, as in ``_build_image_path``.
        """
        root = base_image_dir or self.base_image_dir or self.base_dir
        super().preprocess(output_dir, preprocessor, base_image_dir=root, **kwargs)

    def _build_labels(self) -> None:
        """Merge the 37 binary pathology labels, with snake_case names as in the base contract."""
        lab = self._read_labels()
        if len(lab.columns) <= 1:
            return
        present = [c for c in PATHOLOGY_LABELS if c in lab.columns]
        lab = lab[["study_uid"] + present].drop_duplicates("study_uid")
        lab = lab.rename(columns={c: c.lower().replace(" ", "_") for c in present})
        self.df = self.df.merge(lab.rename(columns={"study_uid": "study_id"}), on="study_id", how="left")

    def _build_report(self) -> None:
        """Merge the report text. It repeats across the series of a study.

        The reports are text, not the file paths that ``BaseHarmonizer._build_report`` expects.
        """
        rep = self._read_reports()
        if rep.empty:
            return
        col = {"full": "report", "findings": "findings",
               "impression": "impression"}[self.report_section]
        keep = ["study_uid"] + [c for c in dict.fromkeys((col, "impression"))
                                if c in rep.columns]
        rep = rep[keep].drop_duplicates("study_uid")
        if col in rep.columns:
            # Copy, then drop. A rename of "impression" also removed the
            # "impression" column and gave 2 "report" columns (V-H2-9).
            rep["report"] = rep[col]
            if col not in ("report", "impression"):
                rep = rep.drop(columns=[col])
        self.df = self.df.merge(rep.rename(columns={"study_uid": "study_id"}), on="study_id", how="left")
