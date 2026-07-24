"""Base dataset for VQA (visual question-answering) over radiology images.

Returns dict samples with keys:
    img            (Tensor; loaded by the MONAI transform)
    question       (str)
    answer         (str)
    answer_struct  (dict; only when output_struct=True and column present)
    question_type  (str; only when output_question_type=True and column present)
    patient_id     (str)
    study_id       (str)
    question_id    (str)

Phase-1 design: single_image_only — one row per (image, question) sample.
Multi-image studies (where one question spans multiple views) are deferred
to Phase 2 along with a custom collate fn.
"""

from __future__ import annotations

import os
import pandas as pd
import torch
import monai as mn
import monai.transforms  # noqa: F401  (ensure mn.transforms is bound regardless of import order)

from radharmony.utils.data_utils import split_data, kfold_splits
from .transforms import _base_load_2d


def _default_vqa_2d_transform(img_size: int = 224, dtype=torch.bfloat16) -> mn.transforms.Compose:
    """Default 2-D image pipeline that preserves non-image dict keys.

    Mirrors :class:`RadiologyTransform2D` for the ``img`` key but omits
    ``SelectItemsD`` so question / answer / id keys flow through unchanged.
    """
    steps = _base_load_2d(["img"])
    steps += [
        mn.transforms.ResizeD(
            keys=["img"], spatial_size=img_size, size_mode="longest", mode="bilinear",
        ),
        mn.transforms.SpatialPadD(
            keys=["img"], spatial_size=(img_size, img_size),
            mode="constant", constant_values=-1,
        ),
        mn.transforms.ToTensorD(keys=["img"], dtype=dtype, track_meta=False),
    ]
    return mn.transforms.Compose(steps)


def _build_vqa_data_dicts(df: pd.DataFrame, base_image_dir: str | None,
                          carry_struct: bool, carry_qtype: bool) -> list[dict]:
    """Build a list of MONAI-ready dicts from a VQA harmonized DataFrame.

    Image paths are joined with *base_image_dir* (when set); object-typed
    columns (e.g. ``answer_struct``) are passed through unchanged so dicts
    survive the trip into the dataset.
    """
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    if base_image_dir:
        sep = "" if base_image_dir.endswith("/") else "/"
        img = base_image_dir + sep + df["image_path"].astype(str)
    else:
        img = df["image_path"].astype(str)

    out = []
    has_struct = carry_struct and "answer_struct" in df.columns
    has_qtype = carry_qtype and "question_type" in df.columns
    for i in range(len(df)):
        row = df.iloc[i]
        d = {
            "img": img.iloc[i],
            "question": str(row["question"]) if pd.notna(row["question"]) else "",
            "answer": str(row["answer"]) if pd.notna(row["answer"]) else "",
            "patient_id": str(row["patient_id"]),
            "study_id": str(row["study_id"]),
            "question_id": str(row["question_id"]),
        }
        if has_struct:
            d["answer_struct"] = row["answer_struct"]
        if has_qtype:
            d["question_type"] = str(row["question_type"]) if pd.notna(row["question_type"]) else ""
        out.append(d)
    return out


class BaseVQADataset:
    """Base class for VQA datasets.  Subclasses implement :meth:`_get_harmonized_df`.

    Args:
        base_image_dir: Root directory prepended to relative ``image_path``.
        transform: MONAI transform pipeline applied to each sample.  Defaults
            to a 2-D radiology pipeline that loads ``img`` only — text fields
            pass through untouched.
        cache_dir: Root directory for MONAI PersistentDataset cache.  Pass
            ``None`` to use an in-memory ``Dataset`` (no caching).
        output_struct: Include ``answer_struct`` (dict) in each sample.
        output_question_type: Include ``question_type`` in each sample.
        harmonized_df: If set, skip harmonization and use this DataFrame.
        harmonizer: Use this harmonizer's ``harmonized_df``.
        harmonizer_path: Load a harmonizer pickle saved via
            :meth:`BaseVQAHarmonizer.save`.  Subclasses must set
            :attr:`_HARMONIZER_CLS`.
    """

    _HARMONIZER_CLS = None

    def __init__(
        self,
        base_image_dir: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_struct: bool = False,
        output_question_type: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        self.base_image_dir = base_image_dir
        self.transform = transform or _default_vqa_2d_transform(dtype=dtype)
        self.cache_dir = cache_dir
        self.output_struct = output_struct
        self.output_question_type = output_question_type
        self._preset_harmonized_df = harmonized_df
        self._user_harmonizer = harmonizer
        self._harmonizer_save_path = harmonizer_path

    # ------------------------------------------------------------------
    # Harmonized DataFrame resolution
    # ------------------------------------------------------------------

    def _get_harmonized_df(self) -> pd.DataFrame:
        raise NotImplementedError

    def _try_resolve_preset_harmonized(self) -> pd.DataFrame | None:
        if self._preset_harmonized_df is not None:
            df = self._preset_harmonized_df
            if not isinstance(df, pd.DataFrame):
                raise TypeError("harmonized_df must be a pandas DataFrame")
            return df.copy()

        if self._harmonizer_save_path:
            hcls = self._HARMONIZER_CLS
            if hcls is None:
                raise ValueError(
                    f"{type(self).__name__} has no _HARMONIZER_CLS; "
                    "cannot use harmonizer_path=."
                )
            h = hcls.load_from_saved(self._harmonizer_save_path)
            return h.harmonized_df

        if self._user_harmonizer is not None:
            return self._user_harmonizer.harmonized_df.copy()

        return None

    def get_harmonized_df(self) -> pd.DataFrame:
        return self._get_harmonized_df()

    def set_harmonized_df(self, df: pd.DataFrame | None) -> None:
        self._preset_harmonized_df = df

    # ------------------------------------------------------------------
    # Dataset construction
    # ------------------------------------------------------------------

    def _make_dataset(self, dicts: list, cache_subdir: str, transform=None):
        t = transform if transform is not None else self.transform
        if self.cache_dir is None:
            return mn.data.Dataset(data=dicts, transform=t)
        return mn.data.PersistentDataset(
            data=dicts,
            transform=t,
            cache_dir=(
                f"{self.cache_dir}/{cache_subdir}" if cache_subdir else self.cache_dir
            ),
        )

    def get_datasets(
        self,
        n_splits: int = None,
        random_state: int = 56,
        train_transform=None,
        val_transform=None,
    ):
        """Return one dataset over the full data, or a (train, val) split."""
        df = self._get_harmonized_df()

        if n_splits is None:
            dicts = _build_vqa_data_dicts(
                df, self.base_image_dir, self.output_struct, self.output_question_type
            )
            return self._make_dataset(dicts, cache_subdir="")

        train_df, val_df = split_data(df, "patient_id", n_splits, random_state)
        train_dicts = _build_vqa_data_dicts(
            train_df, self.base_image_dir, self.output_struct, self.output_question_type
        )
        val_dicts = _build_vqa_data_dicts(
            val_df, self.base_image_dir, self.output_struct, self.output_question_type
        )
        return (
            self._make_dataset(train_dicts, "train", transform=train_transform),
            self._make_dataset(val_dicts, "val", transform=val_transform),
        )

    def get_folds(
        self,
        n_splits: int = 5,
        random_state: int = 56,
        train_transform=None,
        val_transform=None,
    ):
        df = self._get_harmonized_df()
        for fold, (train_df, val_df) in enumerate(
            kfold_splits(df, "patient_id", n_splits, random_state)
        ):
            train_dicts = _build_vqa_data_dicts(
                train_df, self.base_image_dir, self.output_struct, self.output_question_type
            )
            val_dicts = _build_vqa_data_dicts(
                val_df, self.base_image_dir, self.output_struct, self.output_question_type
            )
            yield (
                self._make_dataset(
                    train_dicts, f"fold_{fold}/train", transform=train_transform
                ),
                self._make_dataset(
                    val_dicts, f"fold_{fold}/val", transform=val_transform
                ),
            )

    def verify_images(self, drop_missing: bool = True) -> pd.DataFrame:
        df = self._get_harmonized_df()

        def _exists(p):
            fp = os.path.join(self.base_image_dir, p) if self.base_image_dir else p
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
