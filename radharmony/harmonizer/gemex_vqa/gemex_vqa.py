"""Harmonizer for GEMeX-VQA: a chest X-ray VQA dataset over MIMIC-CXR images.

Four question subtypes across 4 JSONL files:
  - open_ended_question.jsonl:    free-text answers, no choices
  - closed_ended_question.jsonl:  binary Yes./No. answers, no choices
  - single_choice_question.jsonl: single-letter answer (for example "C"), has choices list
  - multi_choice_question.jsonl:  multi-letter answer (for example ["A","B","D"]), has choices list

Source: https://huggingface.co/datasets/BoKelvin/GEMeX-VQA. Download the four
JSONL files.

.. note::
   The files in the local copy are the sample split on HuggingFace (20
   questions for each subtype, 80 in total). The full benchmark has many more.
   Check the HuggingFace page and download again when the full split is out.

Images: MIMIC-CXR-JPG. The image paths follow the MIMIC-CXR-JPG layout, so
``base_image_dir`` is its ``files/`` folder, for example
``/data/mimic-cxr-jpg/2.0.0/files``.

Layout:

    <data_dir>/
      closed_ended_question.jsonl
      open_ended_question.jsonl
      single_choice_question.jsonl
      multi_choice_question.jsonl

    <base_image_dir>/
      p10/p10046166/s50051329/abea5eb9-....jpg
      ...

**Harmonized columns (required):**
  patient_id, study_id, question_id, image_path, question, answer

**Extra columns:**
  answer_struct  (dict: reason, visual_regions, visual_locations, ori_report, choices)
  question_type  (type field: disease, abnormality, finding, size, location, severity)
  question_subtype  (open_ended | closed_ended | single_choice | multi_choice)
  choices        (list of "A: ..." strings, or None for open/closed-ended)
  reason         (rationale string)
  ori_report     (original radiology report)
  visual_regions (list of anatomical region names)
  visual_locations (list of boxes)

The boxes of ``visual_locations`` are ``[x1, y1, x2, y2]`` on a 224 x 224 (H x W) grid. They are not
D-07 fractions.
"""

import json
import os

import pandas as pd

from ..base_vqa import BaseVQAHarmonizer


_SUBTYPES = [
    "open_ended",
    "closed_ended",
    "single_choice",
    "multi_choice",
]

_FILENAME = {
    "open_ended":    "open_ended_question.jsonl",
    "closed_ended":  "closed_ended_question.jsonl",
    "single_choice": "single_choice_question.jsonl",
    "multi_choice":  "multi_choice_question.jsonl",
}


def _parse_ids(image_path: str):
    """Extract (patient_id, study_id) from a GEMeX/MIMIC-CXR image path.

    Path format: ``p{prefix}/p{patient_id}/s{study_id}/{dicom_id}.jpg``
    Returns digits, the official MIMIC ids, like ``"10046166"`` and ``"50051329"`` (D-01, Q-12).
    """
    parts = image_path.replace("\\", "/").split("/")
    # parts: [prefix, patient_id, study_id, filename]
    patient_id = parts[1].lstrip("p") if len(parts) >= 3 else ""
    study_id   = parts[2].lstrip("s") if len(parts) >= 3 else ""
    return patient_id, study_id


def _normalize_answer(raw_answer) -> str:
    """Coerce answer to a plain string (join list answers with commas)."""
    if isinstance(raw_answer, list):
        return ",".join(str(v) for v in raw_answer)
    return str(raw_answer) if raw_answer is not None else ""


class GEMeXVQAHarmonizer(BaseVQAHarmonizer):
    """Harmonize the GEMeX-VQA dataset.

    Args:
        data_dir: Directory containing the 4 GEMeX-VQA JSONL files.
        base_image_dir: Root of the MIMIC-CXR-JPG ``files/`` directory.
            It is used to load images. The harmonized table does not need it.
        question_subtypes: Which subtypes to include.  Default includes all
            four: ``["open_ended", "closed_ended", "single_choice",
            "multi_choice"]``.
    """

    EXTRA_OUTPUT_COLS = [
        "question_subtype",
        "reason",
        "ori_report",
        "visual_regions",
        "visual_locations",
        "choices",
    ]

    def __init__(
        self,
        data_dir: str = None,
        base_image_dir: str = None,
        question_subtypes: list[str] | None = None,
    ):
        super().__init__(base_image_dir=base_image_dir)
        self.data_dir = os.path.expanduser(data_dir) if data_dir else None
        self.question_subtypes = list(question_subtypes) if question_subtypes else list(_SUBTYPES)
        _unknown = set(self.question_subtypes) - set(_SUBTYPES)
        if _unknown:
            raise ValueError(f"Unknown question_subtypes: {_unknown}. Must be in {_SUBTYPES}")

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        if self.data_dir is not None:
            snap["data_dir"] = self.data_dir
        if self.question_subtypes != list(_SUBTYPES):
            snap["question_subtypes"] = self.question_subtypes
        return snap

    def _read_source(self) -> pd.DataFrame:
        """Read the JSONL files of the chosen subtypes into one table, with one row per question."""
        if self.data_dir is None:
            raise ValueError("data_dir is required to harmonize from scratch.")

        rows = []
        for subtype in self.question_subtypes:
            path = os.path.join(self.data_dir, _FILENAME[subtype])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"GEMeX-VQA file not found: {path}\n"
                    f"Download from https://huggingface.co/datasets/BoKelvin/GEMeX-VQA"
                )
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    patient_id, study_id = _parse_ids(rec["image_path"])
                    choices = rec.get("choices")  # list or None
                    answer  = _normalize_answer(rec.get("answer", ""))
                    rows.append({
                        "patient_id":       patient_id,
                        "study_id":         study_id,
                        "question_id":      f"{subtype}_{rec['q_id']}",
                        "image_path":       rec["image_path"],
                        "question":         rec.get("question", ""),
                        "answer":           answer,
                        "answer_struct": {
                            "reason":           rec.get("reason", ""),
                            "visual_regions":   rec.get("visual_regions", []),
                            "visual_locations": rec.get("visual_locations", []),
                            "ori_report":       rec.get("ori_report", ""),
                            "choices":          choices,
                        },
                        "question_type":    rec.get("type", ""),
                        "question_subtype": subtype,
                        "reason":           rec.get("reason", ""),
                        "ori_report":       rec.get("ori_report", ""),
                        "visual_regions":   rec.get("visual_regions", []),
                        "visual_locations": rec.get("visual_locations", []),
                        "choices":          choices,
                    })

        return pd.DataFrame(rows)
