"""VQA evaluator: a question-conditioned VLM scored with the MedGemma protocol.

A thin :class:`GenerativeEvaluator` subclass. The per-sample *context* handed
to the model is the **question**; the *reference* is the dataset's short
**answer** string. Answers are scored (MedGemma report §3.4, Table 9):

- **overall** tokenized F1 across *all* QAs,
- **closed** accuracy on the yes/no subset,
- **open** tokenized F1 + token recall on the remaining QAs.

Wrap any question-conditioned VLM into ``vqa_answerer(imgs, questions) ->
list[str]`` — see ``backbones.make_medgemma_vqa`` (paper replication) /
``backbones.make_chexagent_vqa``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .._registry import register_evaluator
from ..metrics.vqa import (
    extract_yes_no,
    is_yes_no_answer,
    token_recall,
    tokenized_f1,
)
from .base_generative import GenerativeEvaluator


@register_evaluator("vqa")
class VQAEvaluator(GenerativeEvaluator):
    """Answer VQA questions with a VLM and score them (MedGemma protocol).

    Parameters
    ----------
    vqa_answerer :
        Callable ``(imgs, questions: list[str]) -> list[str]`` of length B.
    dataset :
        A VQA dataset (``VQARadDataset`` / ``MIMICExtCXRQBADataset``). Build it
        with the recipe's ``transform`` so ``sample["img"]`` is whatever the
        answerer expects.
    is_closed :
        Predicate ``(reference_answer) -> bool`` marking the closed (yes/no)
        subset. Default: reference is exactly "yes"/"no" (MedGemma).
    max_samples, n_bootstrap, bootstrap_seed, device, batch_size, num_workers,
    output_dir : see :class:`GenerativeEvaluator`.
    """

    _progress_desc = "answering VQA"

    def __init__(
        self,
        vqa_answerer,
        *,
        dataset,
        is_closed=is_yes_no_answer,
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        max_samples: int | None = None,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        output_dir: str | None = None,
    ):
        super().__init__(
            vqa_answerer,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_samples=max_samples,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            output_dir=output_dir,
        )
        self.is_closed = is_closed

    def _prepare_batch(self, batch):
        if "question" not in batch or "answer" not in batch:
            raise KeyError(
                "VQA dataset samples need 'question' and 'answer' keys. Pass a "
                "BaseVQADataset (VQARadDataset, MIMICExtCXRQBADataset) with a "
                "transform that keeps the text fields (no SelectItemsD)."
            )
        questions = list(batch["question"])
        answers = list(batch["answer"])
        # ids == contexts == questions (used as input and as the record key).
        return batch["img"], questions, answers, questions

    def _score(self, refs: list[str], hyps: list[str]) -> list[dict]:
        pairs = list(zip(refs, hyps))
        closed = [(r, h) for r, h in pairs if self.is_closed(r)]
        openq = [(r, h) for r, h in pairs if not self.is_closed(r)]

        rows = [{
            "label": "overall",
            "token_f1": float(np.mean([tokenized_f1(h, r) for r, h in pairs])),
        }]
        if closed:
            rows.append({
                "label": "closed",
                "accuracy": float(np.mean(
                    [extract_yes_no(h) == r.strip().lower() for r, h in closed]
                )),
                "token_f1": float(np.mean([tokenized_f1(h, r) for r, h in closed])),
            })
        if openq:
            rows.append({
                "label": "open",
                "token_f1": float(np.mean([tokenized_f1(h, r) for r, h in openq])),
                "token_recall": float(np.mean([token_recall(h, r) for r, h in openq])),
            })
        return rows

    def _records_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                (q, r, h, bool(self.is_closed(r)))
                for q, r, h in zip(self._ids, self._refs, self._hyps)
            ],
            columns=["question", "reference", "hypothesis", "closed"],
        )
