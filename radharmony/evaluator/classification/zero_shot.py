"""Zero-shot vision-language evaluator.

Scores each test image against a set of per-label text prompts using a
joint image/text embedding space. No training loop — the test set is the
only data used; ``n_seeds`` is ignored. Bootstrap still applies for
test-row CIs.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
import torch

from .._registry import register_evaluator
from ..metrics.classification import compute_metrics, macro_average, METRIC_KEYS
from .base import BaseClsEvaluator


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return X / norms


@register_evaluator("zero_shot")
class ZeroShotEvaluator(BaseClsEvaluator):
    """Zero-shot classification via image-text cosine similarity."""

    def __init__(
        self,
        image_encoder,
        text_encoder: Callable[[list[str]], torch.Tensor],
        prompts: dict[str, list[str]],
        *,
        negative_prompts: dict[str, list[str]] | None = None,
        tokenizer: Callable | None = None,
        dataset=None,
        train_dataset=None,
        test_dataset=None,
        **base_kwargs,
    ):
        # Zero-shot uses only a test set. If the user passes `test_dataset=`
        # alone, route it into the `dataset=` slot so BaseEvaluator's split-mode
        # enforcement accepts it. `train_dataset` is silently dropped.
        if test_dataset is not None and dataset is None:
            dataset = test_dataset
            test_dataset = None
            train_dataset = None

        super().__init__(
            image_encoder,
            dataset=dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            **base_kwargs,
        )
        self.text_encoder = text_encoder
        self.prompts = prompts
        self.negative_prompts = negative_prompts or {}
        self.tokenizer = tokenizer

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        if self.tokenizer is not None:
            # HF joint processors (e.g. SiglipProcessor) put the first positional
            # arg into the image_processor slot; text strings must go via text=.
            if hasattr(self.tokenizer, "image_processor"):
                # SigLIP-family models were trained with a fixed 64-token context;
                # their contract is padding="max_length". padding=True (pad to
                # longest in batch) both warns and yields degraded embeddings.
                if "Siglip" in type(self.tokenizer).__name__:
                    tokens = self.tokenizer(
                        text=texts, return_tensors="pt", padding="max_length"
                    )
                else:
                    tokens = self.tokenizer(text=texts, return_tensors="pt", padding=True)
            else:
                tokens = self.tokenizer(texts)
            out = self.text_encoder(tokens)
        else:
            out = self.text_encoder(texts)
        if isinstance(out, torch.Tensor):
            out = out.float().detach().cpu().numpy()
        return np.asarray(out)

    def _label_prototype(self, label: str) -> tuple[np.ndarray, np.ndarray | None]:
        if label not in self.prompts:
            raise KeyError(f"No prompts registered for label {label!r}.")
        pos = _l2_normalize(self._encode_text(self.prompts[label])).mean(axis=0)
        pos = pos / (np.linalg.norm(pos) + 1e-12)
        neg = None
        if label in self.negative_prompts:
            neg = _l2_normalize(self._encode_text(self.negative_prompts[label])).mean(axis=0)
            neg = neg / (np.linalg.norm(neg) + 1e-12)
        return pos, neg

    def evaluate(self) -> pd.DataFrame:
        # Either `dataset` (user passed it, or we routed `test_dataset` into it)
        # or an explicit `test_dataset` (fixed-split case where train is ignored).
        test_ds = self.test_dataset if self.test_dataset is not None else self.dataset
        X_test, y_test = self._load_or_extract(test_ds, cache_suffix="test")
        X_test = _l2_normalize(X_test)

        prototypes = [self._label_prototype(name) for name in self.active_labels]
        scores = np.zeros_like(y_test, dtype=np.float32)
        for k, (pos, neg) in enumerate(prototypes):
            s = X_test @ pos
            if neg is not None:
                s = s - (X_test @ neg)
            scores[:, k] = s

        rows = []
        point_tmpl = {"n_train": -1, "fold": -1, "seed": -1, "bootstrap": -1}
        for k, name in enumerate(self.active_labels):
            row = dict(point_tmpl, label=name)
            row.update(compute_metrics(y_test[:, k], scores[:, k], self.threshold_strategy))
            rows.append(row)

        if self.n_bootstrap > 0:
            rng = np.random.default_rng(self.bootstrap_seed)
            n = y_test.shape[0]
            for b in range(self.n_bootstrap):
                idx = rng.integers(0, n, size=n)
                tmpl = dict(point_tmpl, bootstrap=b)
                for k, name in enumerate(self.active_labels):
                    row = dict(tmpl, label=name)
                    row.update(
                        compute_metrics(
                            y_test[idx, k], scores[idx, k], self.threshold_strategy
                        )
                    )
                    rows.append(row)

        df = pd.DataFrame(rows)
        return macro_average(df)
