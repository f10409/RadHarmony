"""Classification evaluators over pre-extracted embeddings (datathon26).

Thin subclasses of the standard probe evaluators that default their
``image_encoder`` to :class:`IdentityEncoder`. Paired with
:class:`~radharmony.datathon26.dataset.embedding_dataset.DatathonEmbeddingDataset`
(which serves ``embedding.npy`` under the ``"img"`` key), the sole encoder
forward pass in ``BaseClsEvaluator.extract_embeddings`` becomes a pass-through,
so the pre-extracted embeddings flow straight into the probe. Every feature of
the originals is inherited unchanged — patient-grouped k-fold, multi-seed,
bootstrap, threshold strategies, ``macro_average``, and the embedding cache.

Usage::

    ds = DatathonEmbeddingDataset(embeddings_dir, csv_path="dataset_DS1.csv")
    ev = DatathonLinearProbeEvaluator(dataset=ds, n_folds=5)
    results = ev.evaluate()
"""

from __future__ import annotations

from radharmony.evaluator._registry import register_evaluator
from radharmony.evaluator.classification import (
    KNNProbeEvaluator,
    LinearProbeEvaluator,
    PrototypeProbeEvaluator,
    SVMProbeEvaluator,
)

from ._identity import IdentityEncoder


@register_evaluator("datathon26_linear_probe")
class DatathonLinearProbeEvaluator(LinearProbeEvaluator):
    """Linear probe on pre-extracted embeddings (no encoder call)."""

    def __init__(self, *, dataset=None, image_encoder=None, **kwargs):
        super().__init__(image_encoder or IdentityEncoder(), dataset=dataset, **kwargs)


@register_evaluator("datathon26_knn_probe")
class DatathonKNNProbeEvaluator(KNNProbeEvaluator):
    """k-NN probe on pre-extracted embeddings (no encoder call)."""

    def __init__(self, *, dataset=None, image_encoder=None, **kwargs):
        super().__init__(image_encoder or IdentityEncoder(), dataset=dataset, **kwargs)


@register_evaluator("datathon26_svm_probe")
class DatathonSVMProbeEvaluator(SVMProbeEvaluator):
    """SVM probe on pre-extracted embeddings (no encoder call)."""

    def __init__(self, *, dataset=None, image_encoder=None, **kwargs):
        super().__init__(image_encoder or IdentityEncoder(), dataset=dataset, **kwargs)


@register_evaluator("datathon26_prototype_probe")
class DatathonPrototypeProbeEvaluator(PrototypeProbeEvaluator):
    """Prototype (nearest-centroid) probe on pre-extracted embeddings."""

    def __init__(self, *, dataset=None, image_encoder=None, **kwargs):
        super().__init__(image_encoder or IdentityEncoder(), dataset=dataset, **kwargs)
