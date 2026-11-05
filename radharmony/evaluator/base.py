"""Abstract base class for radharmony evaluators.

It checks the split mode (k-fold or fixed-split), sets up the variance
(n_seeds and n_bootstrap) and saves the results. The bootstrap draw is
``radharmony.evaluator.sampling.bootstrap_indices`` (D-32). It works with
any encoder. Subclasses add the feature extraction for each task.
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


# Sentinel values for columns that aren't meaningful in a given split mode / row.
MISSING_INT = -1


class BaseEvaluator(ABC):
    """Abstract base for all evaluators.

    Subclasses must implement :meth:`evaluate`.

    Split modes (exactly one must be active):

    - **k-fold mode**: pass ``dataset=...``. Variance comes from folds;
      ``n_seeds`` and ``n_bootstrap`` are ignored (warn if set).
    - **fixed-split mode**: pass ``train_dataset=...`` and ``test_dataset=...``.
      Variance comes from TWO sources, composed:
        * ``n_seeds`` train-subsample replicates
        * ``n_bootstrap`` test-row resamples
    """

    def __init__(
        self,
        dataset=None,
        train_dataset=None,
        test_dataset=None,
        output_dir: str | None = None,
        n_seeds: int = 1,
        base_seed: int = 0,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        threshold_strategy: str = "youden",
    ):
        has_single = dataset is not None
        has_pair = train_dataset is not None and test_dataset is not None
        has_partial_pair = (train_dataset is None) ^ (test_dataset is None)

        if has_single and (train_dataset is not None or test_dataset is not None):
            raise ValueError(
                "Pass either `dataset=` (k-fold mode) OR "
                "`train_dataset=` + `test_dataset=` (fixed-split mode), not both."
            )
        if not has_single and not has_pair:
            if has_partial_pair:
                raise ValueError(
                    "Fixed-split mode requires BOTH `train_dataset` and `test_dataset`."
                )
            raise ValueError(
                "Must pass either `dataset=` (k-fold mode) or "
                "`train_dataset=` + `test_dataset=` (fixed-split mode)."
            )

        self.dataset = dataset
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.split_mode = "kfold" if has_single else "fixed"

        if self.split_mode == "kfold" and (n_seeds != 1 or n_bootstrap != 0):
            warnings.warn(
                "`n_seeds` and `n_bootstrap` are ignored in k-fold mode; "
                "variance comes from folds.",
                stacklevel=2,
            )

        self.output_dir = output_dir
        self.n_seeds = n_seeds
        self.base_seed = base_seed
        self.n_bootstrap = n_bootstrap
        self.bootstrap_seed = bootstrap_seed
        self.threshold_strategy = threshold_strategy

    @abstractmethod
    def evaluate(self) -> pd.DataFrame:
        """Run the evaluation and return the per-row metric DataFrame.

        The schema for classification evaluators is::

            label, n_train, fold, seed, bootstrap, <metric panel>

        with ``fold = -1`` in fixed-split mode, ``seed = -1`` in k-fold mode,
        and ``bootstrap = -1`` for point estimates.
        """

    def save_results(self, df: pd.DataFrame, output_dir: str | None = None) -> str:
        """Write per-row CSV and per-label summary CSV to *output_dir*.

        Returns the directory path. Raises ValueError if no output_dir is set.
        """
        out = output_dir if output_dir is not None else self.output_dir
        if out is None:
            raise ValueError(
                "save_results requires `output_dir` (either on the evaluator "
                "or passed to this call)."
            )
        os.makedirs(out, exist_ok=True)

        rows_path = os.path.join(out, "results.csv")
        df.to_csv(rows_path, index=False)

        summary = self._summarize(df)
        summary_path = os.path.join(out, "results_summary.csv")
        summary.to_csv(summary_path, index=False)

        return out

    def _summarize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate per-row results into a per-label summary.

        Groups by ``label`` and ``n_train`` (if present). For each numeric metric
        column (D-19, V-E-15):

        - mean: the mean of the point-estimate rows (``bootstrap == -1``).
        - n: the number of point-estimate rows with a value (Q-8). A label with NaN
          runs warns one time.
        - std and 95% CI: over the bootstrap rows. With no bootstrap rows, over the
          point-estimate rows (the folds or the seeds).
        """
        group_cols = ["label"]
        if "n_train" in df.columns:
            group_cols.append("n_train")

        reserved = {"label", "n_train", "fold", "seed", "bootstrap"}
        metric_cols = [
            c for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not metric_cols:
            return df.groupby(group_cols).size().reset_index(name="n")

        rows, nan_runs = [], {}
        for key, sub in df.groupby(group_cols):
            if not isinstance(key, tuple):
                key = (key,)
            row = dict(zip(group_cols, key, strict=True))
            if "bootstrap" in sub.columns:
                point, boot = sub[sub["bootstrap"] < 0], sub[sub["bootstrap"] >= 0]
            else:
                point, boot = sub, sub.iloc[0:0]
            spread = boot if len(boot) else point
            for m in metric_cols:
                p = point[m].to_numpy(dtype=float)
                row[f"{m}_mean"] = np.nanmean(p) if len(p) else np.nan
                row[f"{m}_n"] = n_ok = int(np.sum(~np.isnan(p)))
                if n_ok < len(p):  # some runs have no value (D-19, D-29)
                    where = f" at n_train {row['n_train']}" if row.get("n_train", -1) != -1 else ""
                    nan_runs.setdefault(row["label"], []).append(f"{m} is NaN in {len(p) - n_ok} of {len(p)} runs{where}")
                vals = spread[m].to_numpy(dtype=float)
                # 1 value has no spread: its std and CI are NaN, not 0.0 (V-E-31).
                if np.sum(~np.isnan(vals)) >= 2:
                    row[f"{m}_std"] = np.nanstd(vals)
                    row[f"{m}_ci_lo"] = np.nanpercentile(vals, 2.5)
                    row[f"{m}_ci_hi"] = np.nanpercentile(vals, 97.5)
                else:
                    row[f"{m}_std"] = row[f"{m}_ci_lo"] = row[f"{m}_ci_hi"] = np.nan
            rows.append(row)

        for label, parts in nan_runs.items():  # one warning per label (Q-8)
            warnings.warn(f"Label {label!r}: {', '.join(parts)}. Their means use the other runs (D-19).",
                          UserWarning, stacklevel=3)
        return pd.DataFrame(rows)
