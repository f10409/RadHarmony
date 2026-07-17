"""Pass-through encoder / generator for pre-computed datathon26 results.

The datathon26 datasets already carry the model outputs (embedding vectors or
generated report text) in the sample's ``"img"`` key. These pass-throughs let
the *unmodified* RadHarmony evaluators consume those outputs: the classification
evaluators call ``image_encoder(imgs)`` once (:class:`IdentityEncoder` returns
the embeddings), and the generative evaluator calls ``generator(imgs, contexts)``
(:func:`identity_generator` returns the predicted reports).
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

    Two-argument signature so ``GenerativeEvaluator._run_generator`` binds it as
    a context-accepting generator; ``contexts`` is ignored.
    """
    return list(imgs)
