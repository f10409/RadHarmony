"""Harmonizer for the VQA-RAD dataset."""

import os

import pandas as pd

from .base_vqa import BaseVQAHarmonizer


class VQARadHarmonizer(BaseVQAHarmonizer):
    """Harmonizer for VQA-RAD (Visual Question Answering in Radiology).

    VQA-RAD has no patient/study concept — each of the 315 MedPix images is
    its own case, so ``patient_id`` and ``study_id`` are both derived from
    the image filename (e.g. ``synpic54610.jpg`` -> ``synpic54610``).
    ``qid`` is globally unique across all 2,248 questions and is used
    directly as ``question_id``.

    ``phrase_type`` in the source JSON marks the dataset's official
    train/test split: ``test_freeform`` / ``test_para`` are the 451-question
    test set used in the VQA-RAD paper; everything else is train. This is
    carried through as ``split``.

    Args:
        json_path: Path to ``VQA_RAD Dataset Public.json``.
        base_image_dir: ``VQA_RAD Image Folder/`` directory (flat, no subdirs).
    """

    EXTRA_OUTPUT_COLS = ["answer_type", "image_organ"]

    def __init__(self, json_path: str, base_image_dir: str = None):
        super().__init__(base_image_dir=base_image_dir)
        self.json_path = json_path

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["json_path"] = self.json_path
        return snap

    def harmonize(self) -> pd.DataFrame:
        df = pd.read_json(self.json_path)

        stem = df["image_name"].apply(lambda p: os.path.splitext(p)[0])
        df["patient_id"] = stem
        df["study_id"] = stem
        df["question_id"] = df["qid"].astype(str)
        df["image_path"] = df["image_name"]
        df["question"] = df["question"].astype(str).str.strip()
        df["answer"] = df["answer"].astype(str).str.strip()
        df["question_type"] = df["question_type"].astype(str)
        # A couple of source rows have a trailing space ("CLOSED ").
        df["answer_type"] = df["answer_type"].astype(str).str.strip()
        df["image_organ"] = df["image_organ"].astype(str)
        # phrase_type in {"freeform", "para"} is train; the "test_" prefixed
        # variants are the paper's held-out 451-question test set.
        df["split"] = df["phrase_type"].apply(
            lambda p: "test" if str(p).startswith("test_") else "train"
        )

        self.df = df.reset_index(drop=True)
        return self._select_harmonized_columns(self.df)
