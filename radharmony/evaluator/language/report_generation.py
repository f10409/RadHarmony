"""Report-generation evaluator: any vision-language model + RadEval metrics.

Mirrors the classification evaluators' design idiom — the evaluator never
inspects model internals. The caller wraps any chest-X-ray VLM (CheXagent,
MedGemma, LLaVA-Rad, MAIRA, …) into one minimal callable:

    report_generator(imgs: Tensor[B, ...]) -> list[str]   # B generated reports

The evaluator runs that generator over a RadHarmony dataset (built with
``output_report=True`` so each sample carries its reference-report path),
extracts the reference section, and scores hypotheses against references
with the RadEval framework (16-metric suite by default).

Split semantics
---------------
Report generation has no training phase, so the k-fold / fixed-split
machinery of :class:`BaseEvaluator` does not apply. The evaluator takes a
single ``dataset=`` (its only role: satisfy the base contract and act as
the evaluation set; folds are never used). Variance, if requested, comes
from bootstrapping the scored (ref, hyp) pairs via this class's own
``n_bootstrap`` — kept separate from the base attribute so the base
class's "ignored in k-fold mode" warning never fires.

Output schema (one row per scoring pass)::

    label, fold, seed, bootstrap, <flat RadEval metric panel>

``label`` is the constant ``"report"`` (no per-label axis exists for
generation), ``fold = seed = -1`` always, ``bootstrap = -1`` for the
point estimate and ``0..n_bootstrap-1`` for resamples. This keeps
:meth:`BaseEvaluator._summarize` (groups by ``label``) working unchanged:
the summary's mean/std/CI are taken across the bootstrap resamples.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .._registry import register_evaluator
from ..base import BaseEvaluator
from ..metrics.language import FULL_METRICS, compute_report_metrics
from ._report_parser import parse_report_section


@register_evaluator("report_generation")
class ReportGenerationEvaluator(BaseEvaluator):
    """Generate reports with a VLM and score them with RadEval.

    Parameters
    ----------
    report_generator :
        Callable ``(imgs: Tensor[B, ...]) -> list[str]`` of length ``B``.
        The evaluator does not touch model internals — wrap whatever VLM
        you have (see ``backbones.make_chexagent_generator``).
    dataset :
        A RadHarmony dataset constructed with ``output_report=True``. Used
        only as the evaluation set (no folds).
    metrics :
        RadEval metric names. Defaults to the full 16-metric suite.
    ref_section :
        Report section to score against — ``"findings"`` (default),
        ``"impression"``, ``"both"``, or ``"full"``.
    max_samples :
        Optional cap on the number of scored pairs (smoke tests / debugging).
    n_bootstrap :
        Number of (ref, hyp) bootstrap resamples for metric CIs. ``0``
        (default) → point estimate only. Note: with the full suite each
        resample re-runs every model-based metric — keep this small.
    """

    def __init__(
        self,
        report_generator,
        *,
        dataset,
        metrics: tuple[str, ...] | list[str] = FULL_METRICS,
        ref_section: str = "findings",
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        max_samples: int | None = None,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        output_dir: str | None = None,
    ):
        # Single-dataset (k-fold) mode satisfies the base contract; folds
        # are never iterated. Base n_bootstrap stays 0 so the base class's
        # k-fold warning never fires — we own bootstrapping below.
        super().__init__(dataset=dataset, output_dir=output_dir)
        self.report_generator = report_generator
        self.metrics = tuple(metrics)
        self.ref_section = ref_section
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self._n_bootstrap = int(n_bootstrap)
        self._bootstrap_seed = int(bootstrap_seed)

        # Populated by evaluate(): list of (sample_id, ref, hyp).
        self.pairs_: list[tuple[str, str, str]] | None = None
        self.n_samples_: int = 0

    # ── generation ────────────────────────────────────────────────────────
    def _generate(self) -> tuple[list[str], list[str], list[str]]:
        """Run the generator over the dataset; return (ids, refs, hyps)."""
        ds = self.dataset
        torch_ds = ds.get_datasets(n_splits=None, num_cores=self.num_workers)
        loader = DataLoader(
            torch_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        ids: list[str] = []
        refs: list[str] = []
        hyps: list[str] = []
        n = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="generating reports", unit="batch"):
                if "report" not in batch:
                    raise KeyError(
                        "Dataset samples have no 'report' key. Construct the "
                        "dataset with output_report=True (and, for MIMIC-CXR, "
                        "pass report_csv_path). MIMIC-CXR-JPG has no reports."
                    )
                imgs = batch["img"]
                report_src = batch["report"]  # list[str] (paths), DataLoader-collated
                generated = self.report_generator(imgs)
                if len(generated) != len(report_src):
                    raise ValueError(
                        f"report_generator returned {len(generated)} reports "
                        f"for a batch of {len(report_src)} images."
                    )
                for src, hyp in zip(report_src, generated):
                    ref = parse_report_section(str(src), self.ref_section)
                    if not ref.strip() or not str(hyp).strip():
                        continue  # unusable reference or empty generation
                    ids.append(str(src))
                    refs.append(ref)
                    hyps.append(str(hyp).strip())
                    n += 1
                    if self.max_samples is not None and n >= self.max_samples:
                        break
                if self.max_samples is not None and n >= self.max_samples:
                    break

        if not refs:
            raise RuntimeError(
                "No usable (reference, hypothesis) pairs were produced — "
                "every reference parsed empty or every generation was blank. "
                f"Check ref_section={self.ref_section!r} and the generator."
            )
        return ids, refs, hyps

    # ── Stage A: generate only (no RadEval import) ────────────────────────
    def generate_only(self, out_parquet: str) -> str:
        """Run the generator, write (sample_id, reference, hypothesis) parquet.

        For the two-stage workflow: this stays import-clean of RadEval so it
        runs in the CheXagent-2 env (transformers==4.40.0). Stage B scores
        the parquet in a separate RadEval env. Returns the parquet path.
        """
        ids, refs, hyps = self._generate()
        self.pairs_ = list(zip(ids, refs, hyps))
        self.n_samples_ = len(refs)
        os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
        pd.DataFrame(
            self.pairs_, columns=["sample_id", "reference", "hypothesis"]
        ).to_parquet(out_parquet, index=False)
        return out_parquet

    # ── evaluation (single-env path) ──────────────────────────────────────
    def evaluate(self) -> pd.DataFrame:
        ids, refs, hyps = self._generate()
        self.pairs_ = list(zip(ids, refs, hyps))
        self.n_samples_ = len(refs)

        def _row(bootstrap: int, panel: dict) -> dict:
            return {
                "label": "report",
                "fold": -1,
                "seed": -1,
                "bootstrap": bootstrap,
                **panel,
            }

        rows = [_row(-1, compute_report_metrics(refs, hyps, self.metrics))]

        if self._n_bootstrap > 0:
            rng = np.random.default_rng(self._bootstrap_seed & 0xFFFFFFFF)
            n = len(refs)
            for b in range(self._n_bootstrap):
                idx = rng.integers(0, n, size=n)
                br = [refs[i] for i in idx]
                bh = [hyps[i] for i in idx]
                rows.append(
                    _row(b, compute_report_metrics(br, bh, self.metrics))
                )

        return pd.DataFrame(rows)

    # ── persistence ───────────────────────────────────────────────────────
    def save_results(
        self, df: pd.DataFrame, output_dir: str | None = None
    ) -> str:
        """Write results.csv + results_summary.csv, plus generations.csv.

        ``generations.csv`` (sample_id, reference, hypothesis) is unique to
        this evaluator — it lets a reader inspect the actual generated text
        behind the aggregate scores.
        """
        out = super().save_results(df, output_dir=output_dir)
        if self.pairs_ is not None:
            gen_path = os.path.join(out, "generations.csv")
            pd.DataFrame(
                self.pairs_, columns=["sample_id", "reference", "hypothesis"]
            ).to_csv(gen_path, index=False)
        return out
