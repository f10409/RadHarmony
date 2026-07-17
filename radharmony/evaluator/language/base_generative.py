"""Shared base for generative (image → text) evaluators.

Report generation and VQA are the *same* underlying task — a vision-language
model produces free text conditioned on an image plus an optional text
prompt, scored against a reference — so they share the whole generation loop,
bootstrap machinery, and persistence. Only three things differ, and those are
the abstract hooks:

- :meth:`_prepare_batch` — pull ``(images, contexts, references, ids)`` out of
  a dataset batch. ``contexts`` is the per-sample text input handed to the
  model (the *question* for VQA; the *indication* — or ``None`` — for report
  generation). ``references`` is the ground truth to score against (the
  *answer* string for VQA; the parsed report section for report generation).
- :meth:`_score` — turn ``(references, hypotheses)`` into metric rows
  (lexical F1 / accuracy for VQA; the RadEval clinical suite for reports).
- :meth:`_records_dataframe` — the ``(inputs, reference, hypothesis)`` table
  written to ``generations.csv`` / the Stage-A parquet.

The generator contract is unified: ``generator(imgs, contexts) -> list[str]``.
A legacy single-argument ``generator(imgs)`` is still supported — the base
detects the accepted arity and calls accordingly.

Split semantics: generation has no training phase, so the base is put in the
base class's single-``dataset`` (k-fold) mode purely to satisfy its contract;
folds are never iterated. Variance comes from this class's own ``n_bootstrap``
over the scored ``(reference, hypothesis)`` pairs.
"""

from __future__ import annotations

import inspect
import os
from abc import abstractmethod

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..base import BaseEvaluator


class GenerativeEvaluator(BaseEvaluator):
    """Base for image→text generative evaluators (report generation, VQA)."""

    _progress_desc = "generating"

    def __init__(
        self,
        generator,
        *,
        dataset,
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        max_samples: int | None = None,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        output_dir: str | None = None,
    ):
        super().__init__(dataset=dataset, output_dir=output_dir)
        self.generator = generator
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_samples = max_samples
        self._n_bootstrap = int(n_bootstrap)
        self._bootstrap_seed = int(bootstrap_seed)

        self._gen_accepts_context: bool | None = None
        self._ids: list | None = None
        self._contexts: list | None = None
        self._refs: list[str] | None = None
        self._hyps: list[str] | None = None
        self.n_samples_: int = 0

    # ── hooks (subclass) ───────────────────────────────────────────────────
    def _torch_dataset(self):
        """Return the iterable torch/MONAI dataset. Override if the dataset's
        ``get_datasets`` needs extra args (report datasets take ``num_cores``)."""
        return self.dataset.get_datasets(n_splits=None)

    @abstractmethod
    def _prepare_batch(self, batch) -> tuple[list, list, list[str], list]:
        """Return ``(images, contexts, references, ids)`` for one batch."""

    @abstractmethod
    def _score(self, refs: list[str], hyps: list[str]) -> list[dict]:
        """Return metric rows (each a dict with a ``label`` + metric columns)."""

    @abstractmethod
    def _records_dataframe(self) -> pd.DataFrame:
        """Return the per-sample ``(inputs, reference, hypothesis)`` table."""

    # ── generation ─────────────────────────────────────────────────────────
    def _run_generator(self, imgs, contexts):
        if self._gen_accepts_context is None:
            try:
                inspect.signature(self.generator).bind(imgs, contexts)
                self._gen_accepts_context = True
            except TypeError:
                self._gen_accepts_context = False
        if self._gen_accepts_context:
            return self.generator(imgs, contexts)
        return self.generator(imgs)  # legacy single-arg generator

    def _generate(self):
        loader = DataLoader(
            self._torch_dataset(),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        ids: list = []
        ctxs: list = []
        refs: list[str] = []
        hyps: list[str] = []
        n = 0
        with torch.no_grad():
            for batch in tqdm(loader, desc=self._progress_desc, unit="batch"):
                b_imgs, b_ctx, b_ref, b_id = self._prepare_batch(batch)
                preds = self._run_generator(b_imgs, b_ctx)
                if len(preds) != len(b_ref):
                    raise ValueError(
                        f"generator returned {len(preds)} outputs for a batch "
                        f"of {len(b_ref)} references."
                    )
                for cid, ctx, ref, hyp in zip(b_id, b_ctx, b_ref, preds):
                    if not str(ref).strip() or not str(hyp).strip():
                        continue  # unusable reference or empty generation
                    ids.append(cid)
                    ctxs.append(ctx)
                    refs.append(str(ref))
                    hyps.append(str(hyp).strip())
                    n += 1
                    if self.max_samples is not None and n >= self.max_samples:
                        break
                if self.max_samples is not None and n >= self.max_samples:
                    break

        if not refs:
            raise RuntimeError(
                "No usable (reference, hypothesis) pairs were produced — every "
                "reference or every generation was blank."
            )
        self._ids, self._contexts, self._refs, self._hyps = ids, ctxs, refs, hyps
        self.n_samples_ = len(refs)
        return ids, ctxs, refs, hyps

    # ── public API ─────────────────────────────────────────────────────────
    def generate_only(self, out_parquet: str) -> str:
        """Run the generator and write the records parquet (no scoring)."""
        self._generate()
        os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
        self._records_dataframe().to_parquet(out_parquet, index=False)
        return out_parquet

    def evaluate(self) -> pd.DataFrame:
        self._generate()

        def _rows(bootstrap: int, rr: list[str], hh: list[str]) -> list[dict]:
            return [
                {"fold": -1, "seed": -1, "bootstrap": bootstrap, **row}
                for row in self._score(rr, hh)
            ]

        rows = _rows(-1, self._refs, self._hyps)
        if self._n_bootstrap > 0:
            rng = np.random.default_rng(self._bootstrap_seed & 0xFFFFFFFF)
            n = len(self._refs)
            for b in range(self._n_bootstrap):
                idx = rng.integers(0, n, size=n)
                rows += _rows(
                    b, [self._refs[i] for i in idx], [self._hyps[i] for i in idx]
                )
        return pd.DataFrame(rows)

    def save_results(self, df: pd.DataFrame, output_dir: str | None = None) -> str:
        """Write results.csv + results_summary.csv, plus generations.csv."""
        out = super().save_results(df, output_dir=output_dir)
        if self._refs is not None:
            self._records_dataframe().to_csv(
                os.path.join(out, "generations.csv"), index=False
            )
        return out
