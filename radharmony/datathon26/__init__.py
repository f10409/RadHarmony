"""RadHarmony datathon26 submodule.

Event-specific datasets and evaluators for the datathon26 CXR benchmark, where
model inference runs on an external *reportbench* service rather than locally.
The service returns per-study results (``embedding.npy`` for classification,
``report.txt`` for report generation); the datasets here serve those results in
place of image tensors, and the evaluators score them by reusing the standard
RadHarmony probe / generative machinery with a pass-through encoder/generator.

Importing this package registers all datathon26 datasets and evaluators in the
shared registries (keys prefixed ``datathon26_``).
"""

from .dataset import (
    DatathonEmbeddingDataset,
    DatathonReportDataset,
    EmbeddingResultsDataset,
    ReportResultsDataset,
    ReportBenchClient,
    perstudy_report_path,
    viewwise_embedding_path,
)
from .evaluator import (
    DatathonKNNProbeEvaluator,
    DatathonLinearProbeEvaluator,
    DatathonPrototypeProbeEvaluator,
    DatathonReportGenerationEvaluator,
    DatathonSVMProbeEvaluator,
    IdentityEncoder,
    identity_generator,
)

__all__ = [
    "DatathonEmbeddingDataset",
    "EmbeddingResultsDataset",
    "DatathonReportDataset",
    "ReportResultsDataset",
    "ReportBenchClient",
    "viewwise_embedding_path",
    "perstudy_report_path",
    "DatathonLinearProbeEvaluator",
    "DatathonKNNProbeEvaluator",
    "DatathonSVMProbeEvaluator",
    "DatathonPrototypeProbeEvaluator",
    "DatathonReportGenerationEvaluator",
    "IdentityEncoder",
    "identity_generator",
]
