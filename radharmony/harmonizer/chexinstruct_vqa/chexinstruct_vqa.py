"""Harmonizer for CheXinstruct's VQA subset (MIMIC-CXR-VQA + MIMIC-Diff-VQA)."""

import json
import re

import pandas as pd

from ..base_vqa import BaseVQAHarmonizer

_STUDY_ID_RE = re.compile(r"\[mimic-cxr\]\s*\[(\d+)\]")
_PATIENT_ID_RE = re.compile(r"/p(\d+)/s\d+/")


class ChexInstructVQAHarmonizer(BaseVQAHarmonizer):
    """Harmonizer for CheXinstruct's MIMIC-CXR-VQA + MIMIC-Diff-VQA records.

    This class gives the train, val and test splits in one table. It is not
    for benchmarks. Use :class:`ChexInstructVQATrainHarmonizer`,
    :class:`ChexInstructVQAValHarmonizer` and
    :class:`ChexInstructVQATestHarmonizer`. Note: the authors did not split
    by patient. 2,007 patients are in more than one split.

    CheXinstruct (Stanford-AIMI/CheXagent) aggregates ~28 source datasets via
    per-source "processor" classes that all emit one unified record schema:
    ``unique_id``, ``study_id``, ``data_source``, ``task_name``,
    ``image_path``, ``text`` (question), ``target_text`` (answer),
    ``qa_pair``, and an optional ``options`` dict for close-ended questions.
    This harmonizer reads a *compiled* JSON produced by running Stanford's
    own (unmodified) ``MIMICCXRVQAProcessor`` and ``MIMICDiffVQAProcessor``
    offline -- see ``docs/wiki/reference/datasets/vqa-text/chexinstruct_vqa.md`` -- and
    translates that schema into RadHarmony's VQA contract. It does not
    embed or adapt Stanford's code (CC BY-NC-ND 4.0); it only consumes data
    those unmodified processors produced.

    ``image_path`` is relative to the image folder of its ``data_source``
    (D-04). Both sources of this dataset are MIMIC-CXR (the source path is
    ``data/mimic-cxr/files/p../p<subject_id>/s<study_id>/<dicom>.jpg``), so
    today one folder, the MIMIC-CXR-JPG ``files/`` folder, holds all images.
    Then ``base_image_dir`` is that folder. With several folders in
    ``source_image_dirs``, ``base_image_dir`` is None, and the dataset class
    joins the folder of each source.

    Args:
        json_path: Path to the compiled ``chexinstruct_vqa.json`` (a flat
            list of records, each carrying a ``split`` field in
            ``{"train", "val", "test"}``).
        base_image_dir: Shorthand for "every configured data source's images
            live under this one root" -- expands to
            ``{src: base_image_dir for src in DATA_SOURCES}``. Both current
            sources (MIMIC-CXR-VQA, MIMIC-Diff-VQA) resolve to MIMIC-CXR-JPG's
            ``files/`` directory, so this is what most callers want. Ignored
            when ``source_image_dirs`` is given.
        source_image_dirs: ``{data_source: root_dir}`` mapping for per-source
            image roots. Takes precedence over ``base_image_dir``. Required
            keys are whichever ``data_source`` values survive the
            ``data_sources`` filter (all of :attr:`DATA_SOURCES` by default).
        question_types: Optional subset of :attr:`QUESTION_TYPES` to keep
            (e.g. ``["Open-Ended VQA"]``, or any combination). ``None`` keeps
            all three task types.
        data_sources: Optional subset of :attr:`DATA_SOURCES` to keep (e.g.
            ``["MIMIC-CXR-VQA"]`` to drop MIMIC-Diff-VQA rows entirely).
            ``None`` keeps both.
    """

    EXTRA_OUTPUT_COLS = ["data_source"]

    # Official split to keep ("train", "val" or "test"). None keeps all splits.
    SPLIT = None

    _IMAGE_PREFIX = "data/mimic-cxr/files/"

    QUESTION_TYPES = frozenset({"Open-Ended VQA", "Close-Ended VQA", "Difference VQA"})
    DATA_SOURCES = frozenset({"MIMIC-CXR-VQA", "MIMIC-Diff-VQA"})

    def __init__(
        self,
        json_path: str,
        base_image_dir: str = None,
        source_image_dirs: dict = None,
        question_types: list[str] = None,
        data_sources: list[str] = None,
    ):
        if source_image_dirs is not None:
            unknown = set(source_image_dirs) - self.DATA_SOURCES
            if unknown:
                raise ValueError(
                    f"Unknown data sources in source_image_dirs {sorted(unknown)}; "
                    f"expected a subset of {sorted(self.DATA_SOURCES)}."
                )
            self.source_image_dirs = dict(source_image_dirs)
        elif base_image_dir is not None:
            self.source_image_dirs = {src: base_image_dir for src in self.DATA_SOURCES}
        else:
            self.source_image_dirs = None
        # One shared folder is the base_image_dir. Several folders give None (D-04).
        roots = set(self.source_image_dirs.values()) if self.source_image_dirs else set()
        super().__init__(base_image_dir=roots.pop() if len(roots) == 1 else None)
        self.json_path = json_path

        if question_types is not None:
            unknown = set(question_types) - self.QUESTION_TYPES
            if unknown:
                raise ValueError(
                    f"Unknown question_types {sorted(unknown)}; "
                    f"expected a subset of {sorted(self.QUESTION_TYPES)}."
                )
        if data_sources is not None:
            unknown = set(data_sources) - self.DATA_SOURCES
            if unknown:
                raise ValueError(
                    f"Unknown data_sources {sorted(unknown)}; "
                    f"expected a subset of {sorted(self.DATA_SOURCES)}."
                )
        self.question_types = question_types
        self.data_sources = data_sources

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["json_path"] = self.json_path
        snap["source_image_dirs"] = self.source_image_dirs
        snap["question_types"] = self.question_types
        snap["data_sources"] = self.data_sources
        return snap

    @staticmethod
    def _first_image_path(value) -> str:
        # MIMIC-Diff-VQA stores a single-element list; MIMIC-CXR-VQA a plain string.
        return value[0] if isinstance(value, list) else value

    def _relative_image_path(self, value) -> str:
        """Return the path relative to the image folder of its source (D-04, Q-11)."""
        return self._first_image_path(value).removeprefix(self._IMAGE_PREFIX)

    def _read_source(self) -> pd.DataFrame:
        with open(self.json_path) as f:
            records = json.load(f)
        df = pd.DataFrame.from_records(records)

        # Filter on the raw columns before the regex steps, which are slow on 700k rows.
        return self.filter_table(df, type_col="task_name")

    def filter_table(self, df: pd.DataFrame, type_col: str = "question_type") -> pd.DataFrame:
        """Keep the rows of ``question_types``, ``data_sources`` and the split of a split class (D-02).

        Args:
            df: The source rows or a harmonized table.
            type_col: The question type column: ``task_name`` in the source, ``question_type`` in a
                harmonized table.

        Returns:
            The kept rows, with a new index.
        """
        if self.question_types is not None:
            df = df[df[type_col].isin(self.question_types)]
        if self.data_sources is not None:
            df = df[df["data_source"].isin(self.data_sources)]
        # A table with no split column is taken as it is.
        if self.SPLIT is not None and "split" in df.columns:
            df = df[df["split"] == self.SPLIT]
        return df.reset_index(drop=True)

    def _build_vqa_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["patient_id"] = df["image_path"].apply(
            lambda p: _PATIENT_ID_RE.search(self._first_image_path(p)).group(1)
        )
        df["study_id"] = df["study_id"].apply(
            lambda s: _STUDY_ID_RE.search(s).group(1)
        )
        df["question_id"] = df["unique_id"].astype(str)
        if self.source_image_dirs is not None:
            missing = sorted(set(df["data_source"]) - set(self.source_image_dirs))
            if missing:
                raise ValueError(
                    f"source_image_dirs has no entry for data_source {missing}. "
                    f"Configured sources: {sorted(self.source_image_dirs)}."
                )
        df["image_path"] = [self._relative_image_path(p) for p in df["image_path"]]
        df["question"] = df["text"].astype(str).str.strip()
        df["answer"] = df["target_text"].astype(str).str.strip()
        df["question_type"] = df["task_name"].astype(str)
        df["answer_struct"] = df["options"] if "options" in df.columns else None
        df["data_source"] = df["data_source"].astype(str)
        # split already present ("train" / "val" / "test") from compilation.
        return df


class ChexInstructVQATrainHarmonizer(ChexInstructVQAHarmonizer):
    """CheXinstruct VQA official train split (``split == "train"``).

    The arguments are the same as :class:`ChexInstructVQAHarmonizer`. The
    two sources made their splits on their own, and not by patient. 497 train patients are also in test, and 1,502
    are also in val.
    """

    SPLIT = "train"


class ChexInstructVQAValHarmonizer(ChexInstructVQAHarmonizer):
    """CheXinstruct VQA official val split (``split == "val"``). See the train class for the shared patients."""

    SPLIT = "val"


class ChexInstructVQATestHarmonizer(ChexInstructVQAHarmonizer):
    """CheXinstruct VQA official test split (``split == "test"``). See the train class for the shared patients."""

    SPLIT = "test"
