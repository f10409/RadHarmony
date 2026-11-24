"""Zero-shot vision-language evaluator.

Scores each test image against a set of per-label text prompts using a
joint image/text embedding space. There is no training loop. The test set is
the only data used, and ``n_seeds`` is ignored. The bootstrap still gives the
CIs. Each sample draws test patients (D-32).

With no validation data, the threshold metrics (F1, accuracy, ...) are NaN
unless ``threshold_strategy="fixed:<value>"`` (D-29). With negative prompts,
``"fixed:0"`` means "closer to the positive prompts".
"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
import pandas as pd
import torch

from .._registry import register_evaluator
from ..metrics.classification import macro_average, pick_threshold
from ._probe_utils import bootstrap_rows, row_patient_ids, score_rows
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

        # The dataset= slot is k-fold mode, where the base class warns that
        # n_bootstrap is ignored. Zero-shot has no folds and uses n_bootstrap,
        # so keep it out of the base class (V-E-14). n_seeds has no use here.
        n_bootstrap = base_kwargs.pop("n_bootstrap", 0)
        n_seeds = base_kwargs.pop("n_seeds", 1)
        if n_seeds != 1:
            warnings.warn("ZeroShotEvaluator ignores n_seeds: it has no train data.", stacklevel=2)
        # Zero-shot always uses cosine similarity, so l2_normalize has no effect (D-17).
        base_kwargs.setdefault("l2_normalize", True)
        if not base_kwargs["l2_normalize"]:
            warnings.warn(
                "ZeroShotEvaluator always uses cosine similarity, so l2_normalize=False has no effect.",
                stacklevel=2,
            )
        super().__init__(
            image_encoder,
            dataset=dataset,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            **base_kwargs,
        )
        self.n_bootstrap = n_bootstrap
        self.text_encoder = text_encoder
        self.prompts = prompts
        self.negative_prompts = negative_prompts or {}
        self.tokenizer = tokenizer
        if not self.threshold_strategy.startswith("fixed:"):
            hint = (
                ' When every label has negative prompts, pass threshold_strategy="fixed:0": a score'
                " above 0 is closer to the positive prompts."
                if self.negative_prompts
                else ' Pass threshold_strategy="fixed:<value>" to get them.'
            )
            warnings.warn(
                "ZeroShotEvaluator has no validation data, so the threshold metrics are NaN "
                "(F1, accuracy, balanced accuracy, TPR, TNR, PPV, NPV, MCC). AUROC and AUPRC "
                "need no threshold (D-29)." + hint,
                stacklevel=2,
            )

    def _warn_long_prompts(self, texts: list[str]) -> None:
        """Warn once for the prompts that are longer than the token length of the model."""
        limit = getattr(getattr(self.tokenizer, "tokenizer", self.tokenizer), "model_max_length", None)
        if not isinstance(limit, int):
            return
        counts = [len(ids) for ids in self.tokenizer(text=texts, padding=False, truncation=False)["input_ids"]]
        long = [t for t, n in zip(texts, counts, strict=True) if n > limit]
        if long:
            warnings.warn(f"{len(long)} prompt(s) are longer than {limit} tokens, so the text encoder uses only "
                          f"the first {limit} tokens. First one: {long[0][:80]!r}", UserWarning, stacklevel=3)

    def _encode_text(self, texts: list[str]) -> np.ndarray:
        if self.tokenizer is not None:
            # HF joint processors (for example SiglipProcessor) put the first
            # positional arg into the image_processor slot. Pass text via text=.
            if hasattr(self.tokenizer, "image_processor"):
                # SigLIP-family models trained with a fixed token context
                # (64 for most, but a recipe may raise it: XraySigLIP uses 256).
                # So pad to the tokenizer's model_max_length. padding=True (pad to
                # longest in batch) warns and gives degraded embeddings.
                if "Siglip" in type(self.tokenizer).__name__:
                    self._warn_long_prompts(texts)
                    # Cut a longer prompt, as the Hugging Face zero-shot pipeline does.
                    tokens = self.tokenizer(
                        text=texts, return_tensors="pt", padding="max_length", truncation=True
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

    @staticmethod
    def _prompts_for(table: dict, label: str) -> list[str] | None:
        """Return the prompts of ``label`` in ``table``, or ``None``.

        The match ignores case, as ``labels=`` does (V-E-22).
        """
        if label in table:
            return table[label]
        by_lower = {str(k).lower(): v for k, v in table.items()}
        return by_lower.get(label.lower())

    def _label_prototype(self, label: str) -> tuple[np.ndarray, np.ndarray | None]:
        pos_prompts = self._prompts_for(self.prompts, label)
        if pos_prompts is None:
            raise KeyError(f"No prompts registered for label {label!r}.")
        pos = _l2_normalize(self._encode_text(pos_prompts)).mean(axis=0)
        pos = pos / (np.linalg.norm(pos) + 1e-12)
        neg = None
        neg_prompts = self._prompts_for(self.negative_prompts, label)
        if neg_prompts is not None:
            neg = _l2_normalize(self._encode_text(neg_prompts)).mean(axis=0)
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

        point_tmpl = {"n_train": -1, "fold": -1, "seed": -1, "bootstrap": -1}
        # Zero-shot has no validation data (D-29). "fixed:<value>" gives that value.
        # Any other strategy gives NaN, so the threshold metrics are NaN.
        thr = [pick_threshold(None, None, self.threshold_strategy)] * len(self.active_labels)
        rows = score_rows(scores, y_test, self.active_labels, thr, self.threshold_strategy, point_tmpl)
        rows += bootstrap_rows(
            scores, y_test, self.active_labels, thr, self.threshold_strategy, point_tmpl,
            self.n_bootstrap, self.bootstrap_seed, groups=row_patient_ids(self, test_ds, len(y_test)),
        )

        df = pd.DataFrame(rows)
        return macro_average(df)
