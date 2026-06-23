"""Convenience wrappers for vision backbones.

``ImageEncoderWrapper`` is a thin ``nn.Module`` adapter that registers a
backbone as a submodule and routes its forward through caller-supplied
``model_call`` / ``pool`` functions. Its purpose is to satisfy the
evaluator contract (``forward(x) -> Tensor[B, D]``) *without* hiding
image preprocessing: channel expansion, resize, and input normalization
are expected to live on the data-pipeline (MONAI transform) side so
they stay encoder-specific and run in DataLoader workers.

Why an ``nn.Module`` instead of a ``lambda``:

- ``self.model`` registers as a submodule, so ``.to(device)``,
  ``state_dict()``, and ``.parameters()`` propagate. A closure over a
  loose ``nn.Module`` silently fails all three — particularly bad for
  ``FinetuneEvaluator``, which calls ``encoder.parameters()`` to
  build the optimizer when ``freeze_backbone=False``.
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn


class ImageEncoderWrapper(nn.Module):
    """Adapt an arbitrary vision backbone to the evaluator contract.

    ``forward(x: Tensor) -> Tensor[B, D]``

    No image preprocessing is done here — ``x`` is passed straight to
    ``model_call`` (or ``model(x)``). Build any channel expansion /
    resize / normalization into your dataset's MONAI transform.

    Parameters
    ----------
    model : nn.Module
        The backbone. Registered as a submodule.
    model_call : callable, optional
        ``(model, x) -> raw_output``. Defaults to ``model(x)``.
        Use this to route through non-default call signatures, e.g.
        ``lambda m, x: m(pixel_values=x)`` for HuggingFace vision models.
    pool : callable, optional
        ``raw_output -> Tensor[B, D]``. Defaults to identity (backbone
        already returns ``(B, D)`` — e.g. ``timm`` with ``num_classes=0``).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        model_call: Callable[[nn.Module, torch.Tensor], Any] | None = None,
        pool: Callable[[Any], torch.Tensor] | None = None,
    ):
        super().__init__()
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"model must be an nn.Module so it can register as a submodule; "
                f"got {type(model).__name__}. If you only have a callable, wrap "
                f"it in a trivial nn.Module subclass."
            )
        self.model = model
        self.model_call = model_call
        self.pool = pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.model_call(self.model, x) if self.model_call is not None else self.model(x)
        return self.pool(raw) if self.pool is not None else raw

    @classmethod
    def from_custom(
        cls,
        model: nn.Module,
        *,
        model_call: Callable[[nn.Module, torch.Tensor], Any] | None = None,
        pool: Callable[[Any], torch.Tensor] | None = None,
    ) -> "ImageEncoderWrapper":
        """Wrap a backbone with caller-supplied ``model_call`` / ``pool`` hooks.

        Equivalent to the base constructor — exposed as a classmethod so
        the non-HuggingFace path is discoverable alongside
        :meth:`from_huggingface`. Use this for ``timm`` (bare wrap when
        ``num_classes=0``), OpenCLIP (``model_call=lambda m, x: m.encode_image(x)``),
        or any bespoke backbone.

        Examples
        --------
        ::

            # timm CNN with num_classes=0 → already returns (B, D), no hooks needed
            import timm
            encoder = ImageEncoderWrapper.from_custom(
                timm.create_model("resnet50", pretrained=True, num_classes=0),
            )

            # OpenCLIP / BiomedCLIP — route through .encode_image
            import open_clip
            clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-16")
            encoder = ImageEncoderWrapper.from_custom(
                clip_model, model_call=lambda m, x: m.encode_image(x),
            )

            # Bespoke backbone returning a dict; pluck the feature tensor out
            encoder = ImageEncoderWrapper.from_custom(
                my_model,
                model_call=lambda m, x: m(x, return_dict=True),
                pool=lambda out: out["features"][:, 0, :],
            )
        """
        return cls(model, model_call=model_call, pool=pool)

    @classmethod
    def from_huggingface(
        cls,
        hf_model: nn.Module,
        *,
        pool: str | Callable = "cls",
    ) -> "ImageEncoderWrapper":
        """Wrap a HuggingFace vision model (ViT, DINO, BEiT, ...).

        Routes the forward through ``model(pixel_values=x)``. ``pool``
        selects how to reduce the output to ``(B, D)``:

        - ``"cls"`` — ``out.last_hidden_state[:, 0]`` (raw CLS token).
        - ``"mean"`` — ``out.last_hidden_state.mean(dim=1)`` (token mean).
        - ``"pooler"`` — ``out.pooler_output`` (the model's learned
          projection of the CLS token, when present; raises if the
          model does not populate it).
        - callable — ``pool(raw_output) -> Tensor[B, D]`` for any
          custom reduction.
        """
        def _call(m, x):
            return m(pixel_values=x)

        def _pooler(out):
            po = getattr(out, "pooler_output", None)
            if po is None:
                raise RuntimeError(
                    "pool='pooler' requested but model output has no "
                    "pooler_output. Use pool='cls' or pool='mean' instead."
                )
            return po

        if pool == "cls":
            _pool = lambda out: out.last_hidden_state[:, 0]
        elif pool == "mean":
            _pool = lambda out: out.last_hidden_state.mean(dim=1)
        elif pool == "pooler":
            _pool = _pooler
        elif callable(pool):
            _pool = pool
        else:
            raise ValueError(
                f"pool must be 'cls', 'mean', 'pooler', or callable, got {pool!r}"
            )
        return cls(hf_model, model_call=_call, pool=_pool)
