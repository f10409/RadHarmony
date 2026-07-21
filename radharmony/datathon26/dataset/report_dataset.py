"""Dataset that pairs reference and predicted reports for the datathon26 event.

For the *report* task, the reportbench inference service writes one
``report.txt`` per study (the model-generated report)::

    <results_dir>/<anon_study_id>/report.txt

This dataset loads that predicted text into the sample's ``"img"`` key while the
**reference** report — the free-text ground truth carried inline in the
anonymized batch frame's ``report`` column — flows into the ``"report"`` key
(via ``output_report=True``). Paired with :func:`identity_generator`, the
existing :class:`ReportGenerationEvaluator` machinery (RadEval metric suite,
bootstrap, ``generations.csv``) scores predicted-vs-reference with no model
call — the generator forward pass becomes a pass-through of the predicted text.
"""

from __future__ import annotations

import pandas as pd
import monai.transforms as mt

from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.registry import register_dataset


def _load_text(path):
    """Read a generated ``report.txt`` (module-level → picklable for workers)."""
    try:
        with open(path, "r", errors="ignore") as f:
            return f.read().strip()
    except OSError:
        return ""


def perstudy_report_path(model: str, name: str = "report", ext: str = ".txt"):
    """Build a ``result_path`` for the real reportbench per-study report layout.

    The service writes one report per study as ``<study_id>/<model>_<name><ext>``
    (e.g. ``.../model-a_report.txt``). Pass the returned function as
    ``result_path=`` to :class:`ReportResultsDataset`.
    """

    def _fn(row):
        return f"{row['study_id']}/{model}_{name}{ext}"

    return _fn


@register_dataset("datathon26_report")
class ReportResultsDataset(BaseRadiologicalDataset):
    """Serve ``(reference report, predicted report, study_id)`` per study.

    Args:
        results_dir: Root directory holding ``<study_id>/report.txt`` per study
            (the reportbench results folder for the report task).
        harmonized_df: Anonymized batch frame (one row per study) with
            ``patient_id`` / ``study_id`` and an inline ``report`` column
            (reference text). Either this or ``csv_path`` is required.
        csv_path: Path to ``dataset_{DS1..DS6}.csv`` (used when ``harmonized_df`` is
            ``None``).
        result_path: Optional callable ``row -> relative path`` mapping each frame
            row to its predicted-report file under ``results_dir``. Defaults to
            ``<study_id>/report.txt``. For the real reportbench layout
            (``<study_id>/<model>_report.txt``) pass :func:`perstudy_report_path`.
        cache_dir: MONAI cache dir. Defaults to ``None`` (in-memory).
    """

    SUPPORTED_OUTPUTS = frozenset({"report"})
    LABEL_COLS: list = []

    def __init__(
        self,
        results_dir: str,
        *,
        harmonized_df: pd.DataFrame | None = None,
        csv_path: str | None = None,
        result_path=None,
        cache_dir: str | None = None,
        transform=None,
    ):
        if harmonized_df is None and csv_path is None:
            raise ValueError("Pass either harmonized_df or csv_path.")
        df = harmonized_df.copy() if harmonized_df is not None else pd.read_csv(csv_path)
        self._batch_df = df
        self._result_path = result_path

        if transform is None:
            transform = mt.Compose(
                [
                    mt.Lambdad(keys="img", func=_load_text),
                    mt.SelectItemsd(keys=["img", "report", "study_id"]),
                ]
            )

        super().__init__(
            base_image_dir=results_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_report=True,
            harmonized_df=df,
        )

    @property
    def _cols(self) -> dict:
        # Carry study_id through as a per-sample id (used for generations.csv).
        cols = super()._cols
        cols["study_id"] = "study_id"
        return cols

    def _get_harmonized_df(self) -> pd.DataFrame:
        df = self._try_resolve_preset_harmonized()
        if df is None:
            raise ValueError(
                "ReportResultsDataset requires a preset harmonized_df/csv_path."
            )
        df = df.copy()
        # Blank out missing references so the evaluator drops them (rather than
        # scoring the literal string "nan").
        if "report" in df.columns:
            df["report"] = df["report"].fillna("").astype(str)
        else:
            df["report"] = ""
        # Point image_path at the predicted report file. Default:
        # <study_id>/report.txt; override via result_path for the real
        # <study_id>/<model>_report.txt layout.
        if self._result_path is not None:
            df["image_path"] = df.apply(self._result_path, axis=1).astype(str)
        else:
            df["image_path"] = df["study_id"].astype(str) + "/report.txt"
        return df
