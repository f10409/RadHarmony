"""Base harmonizer for VQA (visual question-answering) radiology datasets.

VQA datasets pair an image (or a study of images) with a natural-language
question and a structured answer.  The harmonized schema is one row per
question (single-image mode), with stable columns every VQA source must
populate.

Schema (single-image mode):
    patient_id, study_id, question_id, image_path,
    question, answer, answer_struct (optional, dict), question_type,
    quality (str | None), split (optional)
"""

import os
import pickle
import pandas as pd


_VQA_CORE_COLS = frozenset(
    {
        "patient_id",
        "study_id",
        "question_id",
        "image_path",
        "question",
        "answer",
        "answer_struct",
        "question_type",
        "quality",
        "split",
    }
)


_SAVE_VERSION = 1


class BaseVQAHarmonizer:
    """Abstract base class for VQA dataset harmonizers.

    Subclasses populate ``self.df`` with the VQA schema (one row per question)
    inside :meth:`harmonize`.  The harmonizer itself only handles save/load,
    schema selection, and image-existence verification.

    Args:
        base_image_dir: Root directory prepended to relative ``image_path``
            values when verifying or loading images downstream.
    """

    EXTRA_OUTPUT_COLS: list = []

    def __init__(self, base_image_dir: str = None):
        self.base_image_dir = base_image_dir
        self.df: pd.DataFrame | None = None
        self._harmonized_df_override: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # Harmonized DataFrame getter / setter
    # ------------------------------------------------------------------

    def _select_harmonized_columns(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        df = self.df if df is None else df
        if df is None:
            raise ValueError("No DataFrame loaded; call harmonize() first.")
        required = ["patient_id", "study_id", "question_id", "image_path", "question", "answer"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Harmonized DataFrame missing required columns: {missing}")
        ordered = required + [
            c for c in ("answer_struct", "question_type", "quality", "split")
            if c in df.columns
        ]
        ordered += [c for c in self.EXTRA_OUTPUT_COLS if c in df.columns and c not in ordered]
        return df[ordered].copy()

    @property
    def harmonized_df(self) -> pd.DataFrame:
        if self._harmonized_df_override is not None:
            return self._harmonized_df_override.copy()
        if self.df is not None:
            return self._select_harmonized_columns(self.df)
        raise ValueError(
            "No harmonized DataFrame available. Call harmonize(), assign "
            "harmonized_df, or use load_from_saved()."
        )

    @harmonized_df.setter
    def harmonized_df(self, value: pd.DataFrame | None) -> None:
        self._harmonized_df_override = None if value is None else value.copy()

    def set_harmonized_df(self, df: pd.DataFrame | None) -> None:
        self.harmonized_df = df

    # ------------------------------------------------------------------
    # Save / load (pickle)
    # ------------------------------------------------------------------

    def _harmonizer_init_snapshot(self) -> dict:
        """Kwargs needed to reconstruct this harmonizer.  Subclasses extend."""
        snap = {}
        if self.base_image_dir is not None:
            snap["base_image_dir"] = self.base_image_dir
        return snap

    def save(self, path: str) -> None:
        df = self.harmonized_df
        payload = {
            "version": _SAVE_VERSION,
            "harmonizer_module": self.__class__.__module__,
            "harmonizer_class": self.__class__.__qualname__,
            "harmonized_df": df,
            "init": self._harmonizer_init_snapshot(),
        }
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=4)

    @classmethod
    def load_from_saved(cls, path: str, **init_overrides):
        with open(path, "rb") as f:
            payload = pickle.load(f)
        expected = f"{cls.__module__}.{cls.__qualname__}"
        got = f"{payload['harmonizer_module']}.{payload['harmonizer_class']}"
        if got != expected:
            raise ValueError(
                f"Saved harmonizer is {got}, but loading with {expected}. "
                "Instantiate the matching class, e.g. "
                f"{payload['harmonizer_module']}.{payload['harmonizer_class']}"
                ".load_from_saved(path)."
            )
        merged = {**payload["init"], **init_overrides}
        inst = cls(**merged)
        inst.set_harmonized_df(payload["harmonized_df"])
        return inst

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Build ``self.df`` with the VQA schema and return the harmonized slice."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Verify image existence
    # ------------------------------------------------------------------

    def verify_images(self, base_image_dir: str = None, drop_missing: bool = True) -> pd.DataFrame:
        df = self.harmonized_df
        base = base_image_dir if base_image_dir is not None else self.base_image_dir

        def _exists(p):
            fp = os.path.join(base, p) if base else p
            return os.path.isfile(fp)

        from tqdm import tqdm

        tqdm.pandas(desc="Verifying images")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing images "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All image files exist.")
        else:
            print(f"Found {len(missing_df)} rows with missing images (not dropped).")

        return missing_df
