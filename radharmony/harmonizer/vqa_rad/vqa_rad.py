"""Harmonizer for the VQA-RAD dataset."""

import os

import pandas as pd

from ..base_vqa import BaseVQAHarmonizer


class VQARadHarmonizer(BaseVQAHarmonizer):
    """Harmonizer for VQA-RAD (Visual Question Answering in Radiology).

    VQA-RAD has no patient or study. Each MedPix image is one case. ``patient_id`` and
    ``study_id`` are the image file name without the extension (``synpic54610.jpg`` gives
    ``synpic54610``). The folder has 315 images, but ``synpic35356.jpg`` has no question, so
    the table has 314 cases. ``qid`` is unique for all 2,248 questions and is the
    ``question_id``.

    ``phrase_type`` in the JSON marks the official split. ``test_freeform`` and ``test_para`` are
    the 451 test questions of the VQA-RAD paper. All others are train. The result is ``split``.

    Args:
        json_path: Path to ``VQA_RAD Dataset Public.json``.
        base_image_dir: The ``VQA_RAD Image Folder/`` folder (flat, no subfolders).
    """

    EXTRA_OUTPUT_COLS = ["answer_type", "image_organ"]

    def __init__(self, json_path: str, base_image_dir: str = None):
        super().__init__(base_image_dir=base_image_dir)
        self.json_path = json_path

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["json_path"] = self.json_path
        return snap

    def _read_source(self) -> pd.DataFrame:
        return pd.read_json(self.json_path)

    def _build_vqa_columns(self, df: pd.DataFrame) -> pd.DataFrame:
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
        # "freeform" and "para" are train. The "test_" values are the 451 test questions of the paper.
        df["split"] = df["phrase_type"].apply(
            lambda p: "test" if str(p).startswith("test_") else "train"
        )
        return df
