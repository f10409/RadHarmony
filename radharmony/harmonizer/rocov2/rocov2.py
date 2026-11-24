"""Harmonizer for ROCOv2 (Radiology Objects in COntext, version 2).

Source: https://zenodo.org/records/8333645 (HF mirror: eltorio/ROCOv2-radiology). Paper:
Rueckert J, et al. ROCOv2: Radiology Objects in COntext Version 2, an Updated Multimodal Image
Dataset. Sci Data 2024;11:688. License: CC BY-NC-SA-4.0.

The paper gives 79,789 radiology figure and caption pairs from the PMC Open Access subset. The
Hugging Face files have 79,793: 59,962 train, 9,904 validation and 9,927 test. ROCOv2 is the
successor of ROCO, with 35,705 more images and better concept extraction. It is a captioning
set, so it uses the VQA schema, with an empty ``question`` and the caption as ``answer``.

Each image has UMLS concepts (CUIs). The Zenodo release has ``<split>_concepts_manual.csv``
(curated concepts) and ``cui_mapping.csv`` (CUI to name). They give a modality label for each
image that does not depend on the caption text. So you can score a caption-based tagger without
a circular test.

Limit: ROCOv2 has only radiology images and no negatives from other specialties. It measures
modality accuracy and radiology recall, not the false-positive rate.

Layout::

    <base_dir>/
      hf/data/{train,validation,test}-*.parquet   # image bytes, image_id, caption, cui
      cui_mapping.csv                             # CUI to name (Zenodo)
      {train,valid,test}_concepts_manual.csv      # curated CUIs for each image (Zenodo)
      images/<split>/<image_id>.jpg               # made by scripts/materialize_rocov2.py

Columns: patient_id, study_id, question_id, image_path, question, answer. ``modality_source`` is
``manual`` if the manual file lists the image, else ``auto`` (the automatic CUIs of the
parquet). ``patient_id`` is the image ID, because the release has no PMCID for each image.
Several figures can come from one article, so the rows are not independent by patient.
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from ..base_vqa import BaseVQAHarmonizer

_SPLITS = ("train", "validation", "test")
# Zenodo names the validation files "valid_*". The HF parquet uses "validation-*".
_ZENODO_SPLIT = {"train": "train", "validation": "valid", "test": "test"}

MODALITIES = ("ct", "mri", "xray", "ultrasound", "angiography", "pet_spect", "mammography")

# CUI to modality. The names of the CUIs are in cui_mapping.csv.
CUI2MOD = {
    "C0040405": "ct",
    "C0729619": "ct",
    "C1306645": "xray",
    "C1962945": "xray",
    "C0024485": "mri",
    "C0994163": "mri",
    "C0041618": "ultrasound",
    "C0002978": "angiography",
    "C0243032": "angiography",
    "C0032743": "pet_spect",
    "C0040399": "pet_spect",
    "C3472245": "pet_spect",
    "C0024671": "mammography",
}


class ROCOv2Harmonizer(BaseVQAHarmonizer):
    """Harmonize ROCOv2 into the RadHarmony VQA schema."""

    EXTRA_OUTPUT_COLS = ["split", "cuis", "modality", "modality_source", "n_modality_cuis"]

    def __init__(
        self,
        base_dir: str = None,
        base_image_dir: str = None,
        splits: list[str] | None = None,
        modalities: list[str] | None = None,
        require_modality: bool = False,
        manual_concepts_only: bool = True,
    ):
        super().__init__(base_image_dir=base_image_dir)
        self.base_dir = os.path.expanduser(base_dir) if base_dir else None
        self.splits = list(splits) if splits else list(_SPLITS)
        self.modalities = list(modalities) if modalities else None
        self.require_modality = require_modality
        self.manual_concepts_only = manual_concepts_only

        bad = set(self.splits) - set(_SPLITS)
        if bad:
            raise ValueError(f"unknown splits: {sorted(bad)}; valid: {_SPLITS}")
        if self.modalities:
            bad = set(self.modalities) - set(MODALITIES)
            if bad:
                raise ValueError(f"unknown modalities: {sorted(bad)}; valid: {MODALITIES}")

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()  # keeps base_image_dir (V-H2-10)
        for k, default in (("base_dir", None), ("splits", list(_SPLITS)),
                           ("modalities", None), ("require_modality", False),
                           ("manual_concepts_only", True)):
            if getattr(self, k) != default:
                snap[k] = getattr(self, k)
        return snap

    def _modality_for(self, cuis: list[str]) -> tuple[str | None, int]:
        """Return the modality and the number of modalities of the concepts.

        The modality is ``None`` if the concepts name no modality or several. An image with CT
        and PET is both, and picking one would make a label that a test then scores.
        """
        mods = {CUI2MOD[c] for c in cuis if c in CUI2MOD}
        # Take len() first. set.pop() changes the set, and the count would be 0 for each
        # single-modality row.
        n = len(mods)
        return (next(iter(mods)) if n == 1 else None), n

    def _read_source(self) -> pd.DataFrame:
        """Read the parquet and concept files of the chosen splits, apply the filters and return the table."""
        if self.base_dir is None:
            raise ValueError("ROCOv2Harmonizer requires base_dir")

        rows = []
        for split in self.splits:
            pqs = sorted(glob.glob(os.path.join(self.base_dir, "hf", "data", f"{split}-*.parquet")))
            if not pqs:
                raise FileNotFoundError(
                    f"no ROCOv2 parquet for split '{split}' under {self.base_dir}/hf/data. "
                    f"Download eltorio/ROCOv2-radiology first."
                )
            df = pd.concat([pd.read_parquet(p, columns=["image_id", "caption", "cui"])
                            for p in pqs], ignore_index=True)

            man_path = os.path.join(self.base_dir,
                                    f"{_ZENODO_SPLIT[split]}_concepts_manual.csv")
            if self.manual_concepts_only and os.path.isfile(man_path):
                man = pd.read_csv(man_path)
                man.columns = ["image_id", "cuis"]
                man["cuis"] = man["cuis"].fillna("").map(
                    lambda s: [c.strip() for c in str(s).split(";") if c.strip()])
                df = df.merge(man, on="image_id", how="left")
                # An image with no manual row takes the automatic CUIs, so its source is "auto" (V-H2-29).
                src = df["cuis"].notna().map({True: "manual", False: "auto"})
                df["cuis"] = df["cuis"].where(df["cuis"].notna(),
                                              df["cui"].map(lambda x: list(x) if x is not None else []))
            else:
                df["cuis"] = df["cui"].map(lambda x: list(x) if x is not None else [])
                src = "auto"

            mm = df["cuis"].map(self._modality_for)
            df["modality"] = [m for m, _ in mm]
            df["n_modality_cuis"] = [n for _, n in mm]
            df["modality_source"] = src
            df["split"] = split
            rows.append(df)

        return self.filter_table(pd.concat(rows, ignore_index=True))

    def filter_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep the rows of ``splits`` and ``modalities``, and drop the rows with no modality if ``require_modality``.

        It works on the source rows and on a harmonized table, which have the same column names (D-02).

        Args:
            df: The source rows or a harmonized table.

        Returns:
            The kept rows, with a new index.
        """
        # A table with no split column is taken as it is.
        if set(self.splits) != set(_SPLITS) and "split" in df.columns:
            df = df[df["split"].isin(self.splits)]
        if self.modalities:
            df = df[df["modality"].isin(self.modalities)]
        if self.require_modality:
            df = df[df["modality"].notna()]
        return df.reset_index(drop=True)

    def _build_vqa_columns(self, d: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame({
            "patient_id": d["image_id"],
            "study_id": d["image_id"],
            "question_id": d["image_id"],
            "image_path": d["split"] + "/" + d["image_id"] + ".jpg",
            "question": "",
            "answer": d["caption"],
        })
        for c in self.EXTRA_OUTPUT_COLS:
            if c in d.columns:
                out[c] = d[c]
        return out
