"""Harmonizer for MIMIC-Ext-CXR-QBA (question-answer + scene graphs over MIMIC-CXR).

Reads parquet metadata (one row per question/image/study) plus per-study
``qa.json`` files inside ``qa.zip`` to produce a flat per-question harmonized
DataFrame keyed against MIMIC-CXR image paths.
"""

import json
import os
import warnings
import zipfile
import pandas as pd
from tqdm import tqdm

from radharmony.harmonizer.base import BaseHarmonizer
from radharmony.harmonizer.base_vqa import BaseVQAHarmonizer


def _study_qa_member(patient_id: str, study_id: str) -> str:
    """Path to ``sxxxx.qa.json`` inside qa.zip for (patient_id, study_id)."""
    return f"qa/{patient_id[:3]}/{patient_id}/{study_id}.qa.json"


def _image_relpath(patient_id: str, study_id: str, image_id: str, ext: str) -> str:
    """MIMIC-CXR image path relative to the ``files/`` root."""
    return f"{patient_id[:3]}/{patient_id}/{study_id}/{image_id}{ext}"


def _flatten_answer(answers: list) -> str:
    """Join the texts of the ``main_answer`` dicts of a question into one answer string (D-51).

    The ``details`` and ``related_information`` texts stay only in ``answer_struct``.
    """
    return " ".join(
        a["text"] for a in answers if a.get("answer_type") == "main_answer" and a.get("text")
    ).strip()


class MIMICExtCXRQBAHarmonizer(BaseVQAHarmonizer):
    """Harmonizer for the MIMIC-Ext-CXR-QBA dataset.

    Args:
        metadata_dir: Directory containing the parquet metadata files
            (``question_metadata.parquet``, ``question_image_metadata.parquet``,
            ``image_metadata.parquet``). Use one of the export subsets, for example
            ``.../exports/A_frontal/metadata/q1M``.
        qa_zip_path: Path to ``qa.zip`` for question/answer text extraction.
            When ``None``, ``question`` and ``answer`` columns are left empty.
        base_image_dir: MIMIC-CXR ``files/`` root, for example
            ``/data/mimic-cxr/2.1.0/files``.  Used downstream for resolving
            relative ``image_path`` values.
        image_ext: Image file extension: ``".dcm"`` (default) for MIMIC-CXR
            DICOMs or ``".jpg"`` for MIMIC-CXR-JPG.
        max_studies: Cap on the number of unique studies to process.  Useful
            for fast iteration during development; ``None`` (default) reads all.
            In a split class, it counts the studies of that split.

    The ``split`` column is the official split of the patient in
    ``patient_metadata.parquet``. This class mixes the splits, so it is not
    for benchmarks. Use the split classes (D-09).

    ``view_position`` is the ``img.view_position`` of the image in ``image_metadata.parquet``,
    with the shared names PA, AP and LATERAL (D-01). The source has no age and no sex.
    """

    #: The official split that a split class keeps (D-09). None keeps all rows.
    SPLIT: str | None = None

    EXTRA_OUTPUT_COLS = ["view_position"]
    #: The shared 2-D view names of D-01, as for the image harmonizers.
    VIEW_POSITION_NAMES = True

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

    def _read_source(self, study_ids=None, patient_ids=None) -> pd.DataFrame:
        """Read the source and return the table, one row per question.

        ``harmonize(study_ids=..., patient_ids=...)`` passes the ids here. The id
        filter runs before the slow ``qa.zip`` step, so a small set of studies is
        fast. It gives the same rows as a filter after ``harmonize()`` (D-02).

        Args:
            study_ids: Keep only these studies, with or without the ``s`` (Q-12).
            patient_ids: Keep only these patients, with or without the ``p`` (Q-12).
        """
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

        # The official split of each patient (V-H1-29). A split class keeps its own
        # patients before the max_studies cut and before the slow qa.zip step.
        pm_path = os.path.join(self.metadata_dir, "patient_metadata.parquet")
        if self.SPLIT is not None and not os.path.isfile(pm_path):
            raise FileNotFoundError(f"{type(self).__name__} needs {pm_path} for the official split (D-35).")
        if os.path.isfile(pm_path):
            split = pd.read_parquet(pm_path).reset_index()[["patient_id", "split"]]
            df = df.merge(split, on="patient_id", how="left")
            if self.SPLIT is not None:
                df = df[df["split"] == self.SPLIT]

        if self.max_studies is not None:
            keep_studies = (
                df[["patient_id", "study_id"]]
                .drop_duplicates()
                .head(self.max_studies)
            )
            df = df.merge(keep_studies, on=["patient_id", "study_id"], how="inner")
        # After the max_studies cut, so the rows equal a filter after harmonize() (D-02).
        if study_ids is not None:
            df = df[df["study_id"].astype(str).isin({"s" + str(s).lstrip("s") for s in study_ids})]
        if patient_ids is not None:
            df = df[df["patient_id"].astype(str).isin({"p" + str(p).lstrip("p") for p in patient_ids})]

        df = self._add_view_position(df)

        # A list, not DataFrame.apply: apply on an empty table gives a table.
        df["image_path"] = [
            _image_relpath(p, s, i, self.image_ext)
            for p, s, i in zip(df["patient_id"], df["study_id"], df["image_id"], strict=True)
        ]
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

        # MIMIC ids are digits in every harmonizer (D-01, Q-12). image_path keeps the p and s letters.
        df["patient_id"] = df["patient_id"].astype(str).str.lstrip("p")
        df["study_id"] = df["study_id"].astype(str).str.lstrip("s")
        # The source question_id is a template name that repeats in each study (V-H1-30).
        df["question_id"] = df["study_id"] + "_" + df["question_id"].astype(str)
        return df

    def _add_view_position(self, df: pd.DataFrame) -> pd.DataFrame:
        """Join ``img.view_position`` of ``image_metadata.parquet`` on the image of each question (D-01).

        With no file, it warns and gives no ``view_position``.
        """
        path = os.path.join(self.metadata_dir, "image_metadata.parquet")
        if not os.path.isfile(path):
            warnings.warn(f"{path} is missing, so the table has no view_position.", UserWarning, stacklevel=2)
            return df
        views = (pd.read_parquet(path, columns=["img.view_position"]).reset_index()
                 .rename(columns={"img.view_position": "view_position"}))
        df = df.merge(views, on=["patient_id", "study_id", "image_id"], how="left")
        # The same name step as the image harmonizers. It reads only VIEW_POSITION_NAMES of self.
        return BaseHarmonizer._build_view_position_names(self, df)

    # ------------------------------------------------------------------
    # Text extraction from qa.zip
    # ------------------------------------------------------------------

    def _extract_text(self, keys: pd.DataFrame) -> pd.DataFrame:
        """Read the qa.json of each study once. Return a DataFrame of the text for each key."""
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
