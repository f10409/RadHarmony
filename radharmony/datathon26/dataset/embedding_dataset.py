"""Dataset that serves pre-extracted embeddings for the datathon26 event.

The datathon26 workflow submits anonymized studies to an external *reportbench*
inference service; for the *embeddings* task the service writes one
``embedding.npy`` per study inside that study's folder::

    <embeddings_dir>/<anon_study_id>/embedding.npy   # e.g. A_patient_00001_study_00001/

This dataset replaces image decoding with ``np.load`` of that vector, so the
sample's ``"img"`` key carries the pre-extracted feature instead of a decoded
image tensor. Paired with :class:`IdentityEncoder`, the existing classification
evaluators (linear / kNN / SVM / prototype probe) score the embeddings with no
other change — the encoder forward pass becomes a pass-through.

The study/label table is the anonymized batch frame produced by
``datathon26/sample_data.ipynb`` (``dataset_{A..E}.csv``): one row per study,
carrying the anonymized ``patient_id`` / ``study_id`` and the one-hot finding
labels. Because that frame also carries free-text and demographic columns,
label columns are auto-detected as the binary (0/1) numeric columns rather than
"everything that is not a core column" — mirroring the notebook's own
prevalence logic — so string extras never leak into the ``cls`` vector.
"""

from __future__ import annotations

import functools

import numpy as np
import pandas as pd
import torch
import monai.transforms as mt

from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.registry import register_dataset


#: Columns that are never classification labels (harmonized core + common
#: harmonizer extras carried through into the batch CSV).
_NON_LABEL_COLS = frozenset(
    {
        "patient_id",
        "study_id",
        "series_id",
        "image_path",
        "view_position",
        "mask_path",
        "bbox",
        "bbox_labels",
        "report",
        "split",
        "source",
    }
)


def _load_npy(path, dtype=torch.float32):
    """Load a saved embedding vector as a torch tensor (module-level → picklable)."""
    arr = np.load(path)
    return torch.as_tensor(np.asarray(arr), dtype=dtype)


def viewwise_embedding_path(model: str, ext: str = ".npy"):
    """Build a ``result_path`` for the real reportbench per-view embedding layout.

    The service writes one embedding **per view** as
    ``<study_id>/<model>_<view><ext>`` (e.g. ``.../model-a_pa.npy``), where the
    view is the stem of the submitted image (``<study_id>/pa.dcm`` → ``pa``).
    Pass the returned function as ``result_path=`` to
    :class:`EmbeddingResultsDataset` and give one row per image.
    """

    def _fn(row):
        import os

        view = os.path.splitext(os.path.basename(str(row["image_path"])))[0]
        return f"{row['study_id']}/{model}_{view}{ext}"

    return _fn


def _detect_label_cols(df: pd.DataFrame) -> list[str]:
    """Return the binary (0/1) numeric columns — the study-level finding labels.

    Excludes core/metadata columns and any non-numeric or non-binary column
    (free-text ``report``, string demographics, list-valued ``bbox`` …) so the
    ``cls`` vector aggregates real labels only.
    """
    cols = []
    for c in df.columns:
        if c in _NON_LABEL_COLS:
            continue
        s = df[c]
        nonnull = s.dropna()
        if nonnull.empty:
            continue
        if not pd.api.types.is_numeric_dtype(s):
            continue
        if nonnull.isin([0, 1]).all():
            cols.append(c)
    return sorted(cols)


@register_dataset("datathon26_embedding")
class EmbeddingResultsDataset(BaseRadiologicalDataset):
    """Serve one pre-extracted ``embedding.npy`` per study, with finding labels.

    Args:
        embeddings_dir: Root directory holding ``<study_id>/embedding.npy`` per
            study (the reportbench results folder for the embeddings task).
        harmonized_df: Anonymized batch frame (one row per study) with
            ``patient_id`` / ``study_id`` and one-hot label columns. Either this
            or ``csv_path`` is required.
        csv_path: Path to ``dataset_{A..E}.csv`` (used when ``harmonized_df`` is
            ``None``).
        label_cols: Explicit label columns. When ``None`` (default), the binary
            0/1 numeric columns are auto-detected.
        result_path: Optional callable ``row -> relative path`` mapping each frame
            row to its embedding file under ``embeddings_dir``. Defaults to
            ``<study_id>/embedding.npy``. For the real reportbench layout
            (``<study_id>/<model>_<view>.npy``) pass
            :func:`viewwise_embedding_path`.
        cache_dir: MONAI cache dir. Defaults to ``None`` (in-memory) — caching a
            ``np.load`` is pointless.
        dtype: Tensor dtype for the loaded embedding (default ``float32``).
    """

    SUPPORTED_OUTPUTS = frozenset({"cls"})
    LABEL_COLS: list = []

    def __init__(
        self,
        embeddings_dir: str,
        *,
        harmonized_df: pd.DataFrame | None = None,
        csv_path: str | None = None,
        label_cols: list[str] | None = None,
        result_path=None,
        cache_dir: str | None = None,
        dtype: torch.dtype = torch.float32,
        transform=None,
    ):
        if harmonized_df is None and csv_path is None:
            raise ValueError("Pass either harmonized_df or csv_path.")
        df = harmonized_df.copy() if harmonized_df is not None else pd.read_csv(csv_path)

        self.dtype = dtype
        self._result_path = result_path
        # Detect labels up front so the base class does not fall back to its
        # naive "every non-core column is a label" inference (which would sweep
        # in string demographics and crash cls aggregation).
        self.LABEL_COLS = sorted(label_cols) if label_cols else _detect_label_cols(df)

        if transform is None:
            transform = mt.Compose(
                [
                    mt.Lambdad(keys="img", func=functools.partial(_load_npy, dtype=dtype)),
                    mt.ToTensord(keys=["cls"]),
                    mt.SelectItemsd(keys=["img", "cls"]),
                ]
            )

        super().__init__(
            base_image_dir=embeddings_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=True,
            harmonized_df=df,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        df = self._try_resolve_preset_harmonized()
        if df is None:
            raise ValueError(
                "EmbeddingResultsDataset requires a preset harmonized_df/csv_path."
            )
        df = df.copy()
        # Point image_path at the embedding file so get_data_dict joins it to
        # <embeddings_dir>/<image_path>. Default: <study_id>/embedding.npy;
        # override via result_path for the real <study_id>/<model>_<view>.npy layout.
        if self._result_path is not None:
            df["image_path"] = df.apply(self._result_path, axis=1).astype(str)
        else:
            df["image_path"] = df["study_id"].astype(str) + "/embedding.npy"
        return df
