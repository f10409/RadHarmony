"""Report-generation evaluator: any vision-language model + RadEval metrics.

This follows the design of the classification evaluators: the evaluator never
inspects model internals. The caller wraps any chest-X-ray VLM (CheXagent,
MedGemma, LLaVA-Rad, MAIRA, …) into one minimal callable:

    report_generator(imgs: Tensor[B, ...]) -> list[str]   # B generated reports

With ``use_indication=True``, the evaluator also gives the indication of each
report (the reason for the exam) to the generator, as generation input:

    report_generator(imgs, indications: list[str | None]) -> list[str]

The indication is never a scoring target.

The evaluator runs that generator over a RadHarmony dataset. Build the
dataset with ``output_report=True``, so each sample carries its reference
report text. The evaluator extracts the reference section and scores the
hypotheses against the references with RadEval (16 metrics by default).

Split semantics
---------------
Report generation has no training phase, so the k-fold / fixed-split
machinery of :class:`BaseEvaluator` does not apply. The evaluator takes a
single ``dataset=``. It only satisfies the base contract and is the
evaluation set. Folds are never used. Variance, if requested, comes
from this class's own ``n_bootstrap``. It is separate from the base
attribute, so the base warning "ignored in k-fold mode" does not fire.
Each bootstrap sample draws patients and keeps all their (ref, hyp) pairs
(D-32). With no ``patient_id``, it draws pairs.

Output schema (one row per scoring pass)::

    label, fold, seed, bootstrap, <flat RadEval metric panel>

``label`` is the constant ``"report"`` (no per-label axis exists for
generation), ``fold = seed = -1`` always, ``bootstrap = -1`` for the
point estimate and ``0..n_bootstrap-1`` for resamples. This keeps
:meth:`BaseEvaluator._summarize` (groups by ``label``) working unchanged.
The summary mean is the point estimate (the ``bootstrap = -1`` row). The
std and the CI come from the bootstrap rows (D-19).

Hypothesis cleaning
-------------------
``evaluate()`` cleans each generated report with
:func:`~radharmony.evaluator.language.normalize_hypothesis` before scoring,
as the Stage B script does (D-50). ``normalize_hyp=False`` scores the raw text.
``generate_only()`` writes the raw text, so Stage B can choose.
"""

from __future__ import annotations

import inspect
import os
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .._registry import register_evaluator
from ..base import BaseEvaluator
from ..metrics.language import FULL_METRICS, compute_report_metrics
from ..sampling import bootstrap_indices, image_ids, patient_ids_of
from ._report_parser import parse_report_section
from .hypothesis import normalize_hypothesis


@register_evaluator("report_generation")
class ReportGenerationEvaluator(BaseEvaluator):
    """Generate reports with a VLM and score them with RadEval.

    Parameters
    ----------
    report_generator :
        Callable ``(imgs: Tensor[B, ...]) -> list[str]`` of length ``B``.
        The evaluator does not touch model internals. Wrap any VLM
        you have (see ``backbones.make_chexagent_generator``).
    dataset :
        A RadHarmony dataset constructed with ``output_report=True``. Used
        only as the evaluation set (no folds).
    metrics :
        RadEval metric names. Defaults to the full 16-metric suite.
    ref_section :
        Report section to score against: ``"findings"`` (default),
        ``"impression"``, ``"both"``, or ``"full"``.
    max_samples :
        Optional cap on the number of scored pairs (smoke tests / debugging).
    n_bootstrap :
        Number of bootstrap samples for metric CIs. Each sample draws
        patients and keeps all their (ref, hyp) pairs. ``0``
        (default) gives a point estimate only. Note: with the full suite each
        resample re-runs every model-based metric, so keep this small.
    normalize_hyp :
        ``True`` (default): remove the CheXagent grounding tags, boxes, ref
        marks and bold marks from each generated report before scoring
        (D-50). ``False``: score the raw text.
    use_indication :
        ``True``: read the indication section of each reference report. Give it
        to the generator as a second argument (``None`` when the report has
        none). The generator must take 2 arguments. ``False`` (default): the
        generator gets the images only.
    indication_section :
        The report section that holds the indication (default ``"indication"``).
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
        normalize_hyp: bool = True,
        use_indication: bool = False,
        indication_section: str = "indication",
    ):
        # Single-dataset (k-fold) mode satisfies the base contract. Folds
        # are never iterated. Base n_bootstrap stays 0 so the base class's
        # k-fold warning never fires. This class does its own bootstrap below.
        super().__init__(dataset=dataset, output_dir=output_dir)
        if device != "cuda":
            # make_chexagent_generator(device=...) and make_maira2_generator(device=...) place the generator.
            warnings.warn("device is not used: the generator sets its own device. Pass device to the "
                          "backbone recipe.", UserWarning, stacklevel=2)
        if use_indication and ref_section in ("full", "indication"):
            # The reference would hold the indication too, so a model that copies its input gets credit.
            raise ValueError(f"use_indication=True cannot be used with ref_section={ref_section!r}: the "
                             "reference would hold the indication. Use 'findings', 'impression' or 'both'.")
        if use_indication:
            # A generator that cannot take the indications fails here, not after the data loads (D-52).
            try:
                inspect.signature(report_generator).bind(None, None)
            except TypeError as e:
                raise TypeError("use_indication=True needs a report_generator that takes 2 arguments: "
                                "(imgs, indications).") from e
            except ValueError:  # no signature (for example, a C function): let the call decide
                pass
        self.report_generator = report_generator
        self.metrics = tuple(metrics)
        self.ref_section = ref_section
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self._n_bootstrap = int(n_bootstrap)
        self._bootstrap_seed = int(bootstrap_seed)
        self.normalize_hyp = bool(normalize_hyp)
        self.use_indication = bool(use_indication)
        self.indication_section = indication_section

        # Populated by evaluate(): list of (sample_id, ref, hyp).
        self.pairs_: list[tuple[str, str, str]] | None = None
        # One patient id per pair (None when the dataset has no patient_id).
        self.patient_ids_: np.ndarray | None = None
        self.n_samples_: int = 0

    # ── generation ────────────────────────────────────────────────────────
    def _generate(self) -> tuple[list[str], list[str], list[str]]:
        """Run the generator over the dataset and return (ids, refs, hyps)."""
        ds = self.dataset
        torch_ds = ds.get_datasets(n_splits=None, num_cores=self.num_workers)
        loader = DataLoader(
            torch_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        # One id per dataset row (D-46). sample["report"] is the report text, not a path (V-E-26).
        all_ids = image_ids(ds)
        if len(all_ids) != len(torch_ds):  # the table rows do not match the samples: use the row number
            all_ids = [str(i) for i in range(len(torch_ds))]
        # One patient id per dataset row, for the bootstrap by patient (D-32).
        all_pids = patient_ids_of(ds)
        if all_pids is not None and len(all_pids) != len(torch_ds):
            warnings.warn(
                "The patient ids do not match the dataset rows. The bootstrap "
                "draws (reference, hypothesis) pairs, not patients.",
                stacklevel=2,
            )
            all_pids = None
        ids: list[str] = []
        refs: list[str] = []
        hyps: list[str] = []
        pids: list[str] = []
        n = 0
        row = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc="generating reports", unit="batch"):
                if "report" not in batch:
                    raise KeyError(
                        "Dataset samples have no 'report' key. Construct the "
                        "dataset with output_report=True (and, for MIMIC-CXR, "
                        "pass report_csv_path). MIMIC-CXR-JPG has no reports."
                    )
                imgs = batch["img"]
                report_src = batch["report"]  # list[str] (report texts), DataLoader-collated
                if self.use_indication:
                    # The indication is generation input only. An empty one gives None.
                    indications = [parse_report_section(str(s), self.indication_section) or None
                                   for s in report_src]
                    generated = self.report_generator(imgs, indications)
                else:
                    generated = self.report_generator(imgs)
                if len(generated) != len(report_src):
                    raise ValueError(
                        f"report_generator returned {len(generated)} reports "
                        f"for a batch of {len(report_src)} images."
                    )
                for src, hyp in zip(report_src, generated, strict=True):
                    pid = None if all_pids is None else all_pids[row]
                    sample_id = all_ids[row]
                    row += 1
                    ref = parse_report_section(str(src), self.ref_section)
                    if not ref.strip() or not str(hyp).strip():
                        continue  # unusable reference or empty generation
                    pids.append(pid)
                    ids.append(sample_id)
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
        self.patient_ids_ = None if all_pids is None else np.asarray(pids)
        return ids, refs, hyps

    # ── Stage A: generate only (no RadEval import) ────────────────────────
    def generate_only(self, out_parquet: str) -> str:
        """Run the generator, write (sample_id, reference, hypothesis) parquet.

        For the two-stage workflow: this stays import-clean of RadEval so it
        runs in the CheXagent-2 env (transformers==4.40.0). Stage B scores
        the parquet in a separate RadEval env. Returns the parquet path.
        When the dataset has patient ids, the parquet also has a
        ``patient_id`` column, so Stage B can draw patients (D-32).
        """
        ids, refs, hyps = self._generate()
        self.pairs_ = list(zip(ids, refs, hyps, strict=False))
        self.n_samples_ = len(refs)
        os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
        df = pd.DataFrame(self.pairs_, columns=["sample_id", "reference", "hypothesis"])
        if self.patient_ids_ is not None:
            df["patient_id"] = self.patient_ids_
        df.to_parquet(out_parquet, index=False)
        return out_parquet

    # ── evaluation (single-env path) ──────────────────────────────────────
    def evaluate(self) -> pd.DataFrame:
        ids, refs, hyps = self._generate()
        if self.normalize_hyp:  # same cleaning as Stage B (D-50)
            hyps = [normalize_hypothesis(h) for h in hyps]
        self.pairs_ = list(zip(ids, refs, hyps, strict=False))
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

        # Each sample draws patients and keeps all their pairs (D-32).
        samples = bootstrap_indices(
            len(refs), self._n_bootstrap, self._bootstrap_seed, self.patient_ids_
        )
        for b, idx in enumerate(samples):
            br = [refs[i] for i in idx]
            bh = [hyps[i] for i in idx]
            rows.append(_row(b, compute_report_metrics(br, bh, self.metrics)))

        return pd.DataFrame(rows)

    # ── persistence ───────────────────────────────────────────────────────
    def save_results(
        self, df: pd.DataFrame, output_dir: str | None = None
    ) -> str:
        """Write results.csv + results_summary.csv, plus generations.csv.

        ``generations.csv`` (sample_id, reference, hypothesis) is unique to
        this evaluator. It lets a reader see the generated text
        behind the aggregate scores.
        """
        out = super().save_results(df, output_dir=output_dir)
        if self.pairs_ is not None:
            gen_path = os.path.join(out, "generations.csv")
            pd.DataFrame(
                self.pairs_, columns=["sample_id", "reference", "hypothesis"]
            ).to_csv(gen_path, index=False)
        return out
