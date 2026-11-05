"""Pass-through encoder / generator for pre-computed datathon26 results.

The datathon26 datasets already carry the model outputs (embedding vectors or
generated report text) in the sample's ``"img"`` key. These pass-throughs let
the unmodified RadHarmony evaluators use those outputs. A classification
evaluator calls ``image_encoder(imgs)``, and :class:`IdentityEncoder` returns
the embeddings. A generative evaluator calls ``generator(imgs, contexts)``, and
:func:`identity_generator` returns the predicted reports.
"""

from __future__ import annotations

import torch


class IdentityEncoder(torch.nn.Module):
    """Return input embeddings unchanged.

    An ``nn.Module`` (not a bare lambda) so ``BaseClsEvaluator._freeze_encoder``
    can call ``.eval()`` / ``.to(device)`` / freeze parameters on it safely.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return x


def identity_generator(imgs, contexts=None):
    """Return the pre-computed predicted reports unchanged.

    It takes 2 arguments, so ``GenerativeEvaluator._run_generator`` gives it the
    ``contexts`` (the indications or questions). It does not use them.
    """
    return list(imgs)
