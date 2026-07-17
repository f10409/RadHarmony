"""Report-generation evaluator: any VLM + RadEval metrics.

A thin :class:`GenerativeEvaluator` subclass. Report generation is the same
image→text task as VQA, with a fixed "write the findings" instruction (baked
into the generator recipe) rather than a per-sample question. The optional
per-sample *context* here is the **indication** (the clinical reason for the
exam) — fair generation input (it is what the radiologist has before
dictating), not a scoring target. The *reference* is the requested report
section parsed from ``sample["report"]``; hypotheses are scored with the
RadEval framework (16-metric suite by default).

The caller wraps any chest-X-ray VLM into::

    report_generator(imgs, indications=None) -> list[str]   # B reports

``indications`` is a list aligned with ``imgs`` (``None`` entries → no
indication). A legacy ``report_generator(imgs)`` still works — the base
detects the arity. See ``backbones.make_chexagent_generator`` /
``backbones.make_maira2_generator``.

Output schema (one row per scoring pass)::

    label, fold, seed, bootstrap, <flat RadEval metric panel>

``label`` is the constant ``"report"``; ``fold = seed = -1``; ``bootstrap =
-1`` for the point estimate, ``0..n_bootstrap-1`` for resamples.
"""

from __future__ import annotations

import pandas as pd

from .._registry import register_evaluator
from ..metrics.language import FULL_METRICS, compute_report_metrics
from ._report_parser import parse_report_section
from .base_generative import GenerativeEvaluator


@register_evaluator("report_generation")
class ReportGenerationEvaluator(GenerativeEvaluator):
    """Generate reports with a VLM and score them with RadEval.

    Parameters
    ----------
    report_generator :
        Callable ``(imgs, indications=None) -> list[str]``. The evaluator does
        not touch model internals — wrap whatever VLM you have (see
        ``backbones.make_chexagent_generator``).
    dataset :
        A RadHarmony dataset constructed with ``output_report=True``.
    metrics :
        RadEval metric names. Defaults to the full 16-metric suite.
    ref_section :
        Report section to score against — ``"findings"`` (default),
        ``"impression"``, ``"both"``, or ``"full"``.
    use_indication :
        When ``True``, parse the indication section from each report and pass
        it to the generator as the per-sample context (MedGemma / MAIRA-2
        indication-conditioned reporting). Default ``False`` (no indication).
    indication_section :
        Section name handed to the report parser for the indication text
        (default ``"indication"``).
    max_samples, n_bootstrap, bootstrap_seed, device, batch_size, num_workers,
    output_dir : see :class:`GenerativeEvaluator`.
    """

    _progress_desc = "generating reports"

    def __init__(
        self,
        report_generator,
        *,
        dataset,
        metrics: tuple[str, ...] | list[str] = FULL_METRICS,
        ref_section: str = "findings",
        use_indication: bool = False,
        indication_section: str = "indication",
        device: str = "cuda",
        batch_size: int = 8,
        num_workers: int = 4,
        max_samples: int | None = None,
        n_bootstrap: int = 0,
        bootstrap_seed: int = 0,
        output_dir: str | None = None,
    ):
        super().__init__(
            report_generator,
            dataset=dataset,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_samples=max_samples,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            output_dir=output_dir,
        )
        self.metrics = tuple(metrics)
        self.ref_section = ref_section
        self.use_indication = use_indication
        self.indication_section = indication_section

    def _torch_dataset(self):
        # Report datasets are BaseRadiologicalDataset — get_datasets takes
        # num_cores (VQA datasets do not).
        return self.dataset.get_datasets(n_splits=None, num_cores=self.num_workers)

    def _prepare_batch(self, batch):
        if "report" not in batch:
            raise KeyError(
                "Dataset samples have no 'report' key. Construct the dataset "
                "with output_report=True (and, for MIMIC-CXR, pass "
                "report_csv_path). MIMIC-CXR-JPG has no reports."
            )
        report_src = batch["report"]  # list[str] paths, DataLoader-collated
        ids = [str(s) for s in report_src]
        refs = [parse_report_section(str(s), self.ref_section) for s in report_src]
        if self.use_indication:
            contexts = [
                parse_report_section(str(s), self.indication_section) or None
                for s in report_src
            ]
        else:
            contexts = [None] * len(report_src)
        return batch["img"], contexts, refs, ids

    def _score(self, refs: list[str], hyps: list[str]) -> list[dict]:
        return [{"label": "report", **compute_report_metrics(refs, hyps, self.metrics)}]

    def _records_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            list(zip(self._ids, self._refs, self._hyps)),
            columns=["sample_id", "reference", "hypothesis"],
        )
