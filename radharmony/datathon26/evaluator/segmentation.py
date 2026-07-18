"""Segmentation evaluators over pre-extracted patch features (datathon26).

Thin subclasses of the standard frozen-feature segmentation probes that default
their ``image_encoder`` to :class:`IdentityEncoder`. Paired with
:class:`~radharmony.datathon26.dataset.seg_dataset.PatchSegResultsDataset`
(which serves the reportbench ``patches`` folded to a dense ``[D, H, W]`` map
under the ``"img"`` key), the encoder forward in
``BaseSegEvaluator`` becomes a pass-through, so the pre-extracted patch features
flow straight into the segmentation head (1×1 conv / conv block / UPerNet) and
are scored with Dice/IoU. Every feature of the originals is inherited unchanged —
patient-grouped k-fold, early stopping, bootstrap, ``macro_average``, prediction
dumps, and head save/load.

Usage::

    ds = PatchSegResultsDataset(out_dir, harmonized_df=df,
                                result_path=viewwise_embedding_path("model-a"),
                                mask_size=224)
    ev = DatathonLinearProbeSegEvaluator(dataset=ds, num_classes=2, n_folds=5)
    results = ev.evaluate()          # per-class + macro_average Dice/IoU/...
"""

from __future__ import annotations

from radharmony.evaluator._registry import register_evaluator
from radharmony.evaluator.segmentation import (
    ConvProbeSegEvaluator,
    LinearProbeSegEvaluator,
    UPerNetSegEvaluator,
)

from ._identity import IdentityEncoder


@register_evaluator("datathon26_linear_probe_seg")
class DatathonLinearProbeSegEvaluator(LinearProbeSegEvaluator):
    """1×1-conv segmentation probe on pre-extracted patch features (no encoder call)."""

    def __init__(self, *, dataset=None, image_encoder=None, num_classes: int = 2, **kwargs):
        super().__init__(
            image_encoder or IdentityEncoder(), dataset=dataset, num_classes=num_classes, **kwargs
        )


@register_evaluator("datathon26_conv_probe_seg")
class DatathonConvProbeSegEvaluator(ConvProbeSegEvaluator):
    """Conv-block segmentation probe on pre-extracted patch features (no encoder call)."""

    def __init__(self, *, dataset=None, image_encoder=None, num_classes: int = 2, **kwargs):
        super().__init__(
            image_encoder or IdentityEncoder(), dataset=dataset, num_classes=num_classes, **kwargs
        )


@register_evaluator("datathon26_upernet_seg")
class DatathonUPerNetSegEvaluator(UPerNetSegEvaluator):
    """UPerNet segmentation probe on pre-extracted patch features (no encoder call)."""

    def __init__(self, *, dataset=None, image_encoder=None, num_classes: int = 2, **kwargs):
        super().__init__(
            image_encoder or IdentityEncoder(), dataset=dataset, num_classes=num_classes, **kwargs
        )
