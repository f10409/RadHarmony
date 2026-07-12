"""Harmonizer for MIMIC-Ext-CXR-QBA (question-answer + scene graphs over MIMIC-CXR).

Reads parquet metadata (one row per question/image/study) plus per-study
``qa.json`` files inside ``qa.zip`` to produce a flat per-question harmonized
DataFrame keyed against MIMIC-CXR image paths.
"""

import json
import zipfile
import pandas as pd
from tqdm import tqdm

from radharmony.harmonizer.base_vqa import BaseVQAHarmonizer


def _study_qa_member(patient_id: str, study_id: str) -> str:
    """Path to ``sxxxx.qa.json`` inside qa.zip for (patient_id, study_id)."""
    return f"qa/{patient_id[:3]}/{patient_id}/{study_id}.qa.json"


def _image_relpath(patient_id: str, study_id: str, image_id: str, ext: str) -> str:
    """MIMIC-CXR image path relative to the ``files/`` root."""
    return f"{patient_id[:3]}/{patient_id}/{study_id}/{image_id}{ext}"


def _flatten_answer(answers: list) -> str:
    """Flatten a list of top-level answer dicts into a single answer string."""
    return " ".join(
        a.get("text", "") for a in answers if a.get("text")
    ).strip()


class MIMICExtCXRQBAHarmonizer(BaseVQAHarmonizer):
    """Harmonizer for the MIMIC-Ext-CXR-QBA dataset.

    Args:
        metadata_dir: Directory containing the parquet metadata files
            (``question_metadata.parquet``, ``question_image_metadata.parquet``,
            ``image_metadata.parquet``).  Use one of the export subsets, e.g.
            ``.../exports/A_frontal/metadata/q1M``.
        qa_zip_path: Path to ``qa.zip`` for question/answer text extraction.
            When ``None``, ``question`` and ``answer`` columns are left empty.
        base_image_dir: MIMIC-CXR ``files/`` root, e.g.
            ``/data/mimic-cxr/2.1.0/files``.  Used downstream for resolving
            relative ``image_path`` values.
        image_ext: Extension of image files on disk — ``".dcm"`` (default) for
            MIMIC-CXR DICOMs or ``".jpg"`` for MIMIC-CXR-JPG.
        max_studies: Cap on the number of unique studies to process.  Useful
            for fast iteration during development; ``None`` (default) reads all.
    """

    def __init__(
        self,
        metadata_dir: str = None,
        qa_zip_path: str = None,
        base_image_dir: str = None,
        image_ext: str = ".dcm",
        max_studies: int | None = None,
    ):
        super().__init__(base_image_dir=base_image_dir)
        self.metadata_dir = metadata_dir
        self.qa_zip_path = qa_zip_path
        self.image_ext = image_ext
        self.max_studies = max_studies

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        for name in ("metadata_dir", "qa_zip_path", "image_ext", "max_studies"):
            v = getattr(self, name, None)
            if v is not None:
                snap[name] = v
        return snap

    # ------------------------------------------------------------------
    # Harmonize
    # ------------------------------------------------------------------

    def harmonize(self) -> pd.DataFrame:
        if self.metadata_dir is None:
            raise ValueError("metadata_dir is required to harmonize from scratch.")

        q_df = pd.read_parquet(f"{self.metadata_dir}/question_metadata.parquet")
        qi_df = pd.read_parquet(f"{self.metadata_dir}/question_image_metadata.parquet")

        q_df = q_df.reset_index()
        qi_df = qi_df.reset_index()

        # Single-image mode: pick one image_id per (patient, study, question).
        # Sort then drop_duplicates for determinism.
        qi_one = (
            qi_df.sort_values(["patient_id", "study_id", "question_id", "image_id"])
                 .drop_duplicates(subset=["patient_id", "study_id", "question_id"], keep="first")
                 [["patient_id", "study_id", "question_id", "image_id"]]
        )

        df = q_df.merge(
            qi_one, on=["patient_id", "study_id", "question_id"], how="inner"
        )

        if self.max_studies is not None:
            keep_studies = (
                df[["patient_id", "study_id"]]
                .drop_duplicates()
                .head(self.max_studies)
            )
            df = df.merge(keep_studies, on=["patient_id", "study_id"], how="inner")

        df["image_path"] = df.apply(
            lambda r: _image_relpath(r["patient_id"], r["study_id"], r["image_id"], self.image_ext),
            axis=1,
        )
        df["question_type"] = df["question.question_type"].astype(str)
        df["quality"] = df["question.quality.rating"].astype(str)

        if self.qa_zip_path:
            text_df = self._extract_text(df[["patient_id", "study_id", "question_id"]])
            df = df.merge(text_df, on=["patient_id", "study_id", "question_id"], how="left")
        else:
            df["question"] = ""
            df["answer"] = ""
            df["answer_struct"] = None

        df["question"] = df["question"].fillna("")
        df["answer"] = df["answer"].fillna("")

        self.df = df.reset_index(drop=True)
        return self._select_harmonized_columns(self.df)

    # ------------------------------------------------------------------
    # Text extraction from qa.zip
    # ------------------------------------------------------------------

    def _extract_text(self, keys: pd.DataFrame) -> pd.DataFrame:
        """Read each unique study's qa.json once; return a (keys → text) DataFrame."""
        wanted = {
            (r["patient_id"], r["study_id"], r["question_id"])
            for _, r in keys.iterrows()
        }
        studies = keys[["patient_id", "study_id"]].drop_duplicates().to_records(index=False)

        rows: list[dict] = []
        with zipfile.ZipFile(self.qa_zip_path) as z:
            for patient_id, study_id in tqdm(studies, desc="Extracting QA text"):
                member = _study_qa_member(patient_id, study_id)
                try:
                    raw = z.read(member)
                except KeyError:
                    continue
                doc = json.loads(raw)
                for q in doc.get("questions", []):
                    qid = q.get("question_id")
                    if (patient_id, study_id, qid) not in wanted:
                        continue
                    answers = q.get("answers", [])
                    rows.append(
                        {
                            "patient_id": patient_id,
                            "study_id": study_id,
                            "question_id": qid,
                            "question": q.get("question", ""),
                            "answer": _flatten_answer(answers),
                            "answer_struct": answers,
                        }
                    )
        return pd.DataFrame(
            rows,
            columns=["patient_id", "study_id", "question_id", "question", "answer", "answer_struct"],
        )
