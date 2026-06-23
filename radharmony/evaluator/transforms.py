"""Evaluator-side MONAI transforms.

This module ships two pieces that plug a foundation model's canonical
preprocessing into a RadHarmony dataset pipeline:

* :class:`EncoderPreprocessTransform` — a thin ``MapTransform`` that calls
  ``preprocess(sample[key])`` and nothing else. The caller's ``preprocess``
  owns the full contract (reading, resizing, normalizing). Use
  :meth:`~EncoderPreprocessTransform.from_huggingface` for HF processors
  and :meth:`~EncoderPreprocessTransform.from_custom` for anything else.
* :class:`RadiologyEncoderTransform` — a fluent builder that parallels
  :class:`radharmony.dataset.transforms.RadiologyTransform2D`, but uses
  :class:`EncoderPreprocessTransform` as the preprocessing step so
  augmentations can still be layered on via ``.with_*`` hooks.

Reading images inside the transform
-----------------------------------

``EncoderPreprocessTransform.from_huggingface`` / ``.from_custom`` accept
a ``reader`` kwarg so the encoder's native file reader runs *here*,
bypassing any MONAI ``LoadImaged`` upstream.
That matters because each encoder was pretrained on a specific reader
(``PIL.Image.open`` for HF/timm/OpenCLIP; a DICOM/NIfTI reader for
volumetric CXR models) and mixing readers silently perturbs pixel
statistics.

- ``reader="pil"`` — ``PIL.Image.open(path).convert("RGB")``. Matches how
  HF ViT checkpoints were pretrained.
- ``reader="monai"`` — ``mt.LoadImage(ensure_channel_first=True)`` +
  a squeeze of any trailing frame dim (DICOM 2D loads as
  ``(C, W, H, 1)``) + ``mt.Transpose([0, 2, 1])`` to fix MONAI's
  ``(C, W, H)`` axis order for 2D images + 1→3 channel expansion so
  HF vision processors (RAD-DINO, ViT, ...) get the 3-channel input
  they normalize. Useful for DICOM / NIfTI.
- ``reader=<callable>`` — your own ``(path) -> image-like``.
- ``reader=None`` — the upstream dataset already loaded the image; skip
  the read step and pass the value straight to the processor.
"""

from __future__ import annotations

import os
from typing import Any, Callable

import monai.transforms as mt
import torch
from monai.config import KeysCollection
from PIL import Image

from radharmony.dataset.transforms import _RadiologyTransformBase


def _resolve_reader(reader: Any) -> Callable[[str], Any] | None:
    """Resolve a ``reader=`` keyword into a concrete ``(path) -> image`` callable."""
    if reader is None:
        return None
    if reader == "pil":
        return lambda p: Image.open(p).convert("RGB")
    if reader == "monai":
        # DICOM 2D loads as (C, W, H, 1) — squeeze the trailing frame dim.
        # 2D NIfTI / PNG via MONAI loads as (C, W, H) — no squeeze needed.
        # Then transpose (C, W, H) -> (C, H, W) for standard downstream shape.
        # Finally expand 1-channel grayscale -> 3-channel, since HF vision
        # processors (RAD-DINO, ViT, ...) normalize with 3-channel mean/std
        # and reject 1-channel tensors.
        def _squeeze_frame_dim(x):
            if x.ndim == 4 and x.shape[-1] == 1:
                return x.squeeze(-1)
            return x

        def _to_3channel(x):
            if x.ndim == 3 and x.shape[0] == 1:
                return x.repeat(3, 1, 1)
            return x

        loader = mt.Compose(
            [
                mt.LoadImage(image_only=True, ensure_channel_first=True),
                mt.Lambda(_squeeze_frame_dim),
                mt.Transpose(indices=[0, 2, 1]),
                mt.Lambda(_to_3channel),
            ]
        )
        return lambda p: loader(p)
    if callable(reader):
        return reader
    raise ValueError(
        f"reader must be 'pil' | 'monai' | callable | None, got {reader!r}"
    )


class EncoderPreprocessTransform(mt.MapTransform):
    """Wrap an encoder-specific preprocessor as a MONAI ``MapTransform``.

    The base class is a pass-through: it calls ``self.preprocess(value)``
    on each configured key and does not convert, read, or otherwise
    touch the input. The wrapped callable owns the full
    ``sample value -> Tensor[C, H, W]`` contract.

    Parameters
    ----------
    keys : KeysCollection
        Dictionary keys this transform applies to (typically ``["img"]``).
    preprocess : callable
        ``sample value -> torch.Tensor[C, H, W]``. Receives whatever
        upstream transforms placed at ``sample[key]`` (a path string, a
        tensor, a PIL image, ...).
    allow_missing_keys : bool
        Passed through to :class:`monai.transforms.MapTransform`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        preprocess: Callable[[Any], Any],
        *,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.preprocess = preprocess

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        for key in self.key_iterator(d):
            d[key] = self.preprocess(d[key])
        return d

    @classmethod
    def from_custom(
        cls,
        keys: KeysCollection,
        preprocess: Callable[[Any], Any],
    ) -> "EncoderPreprocessTransform":
        """Wrap an arbitrary ``preprocess`` callable.

        ``preprocess`` owns the full contract — reading (if the input is a
        path), resizing, normalizing, and returning a ``Tensor[C, H, W]``.
        Equivalent to the base constructor; exposed as a classmethod so the
        non-HuggingFace path is discoverable alongside
        :meth:`from_huggingface`.

        Parameters
        ----------
        keys : KeysCollection
        preprocess : callable
            ``(path | image-like) -> torch.Tensor[C, H, W]``.
        """
        return cls(keys, preprocess=preprocess)

    @classmethod
    def from_huggingface(
        cls,
        keys: KeysCollection,
        processor: str | Any,
        *,
        reader: Any = "pil",
    ) -> "EncoderPreprocessTransform":
        """Wrap a HuggingFace ``AutoImageProcessor``.

        Parameters
        ----------
        keys : KeysCollection
        processor : str or AutoImageProcessor
            Model id on the Hub (``"microsoft/rad-dino"``) or an
            already-constructed processor instance. To change the
            resize / crop size, construct the processor yourself with
            ``AutoImageProcessor.from_pretrained(..., size=...)`` — the
            schema (``shortest_edge`` vs ``height``/``width`` vs
            ``longest_edge``) varies by model family.
        reader : {'pil', 'monai', callable, None}, default 'pil'
            How to load the image when ``sample[key]`` is a path string.
            ``None`` disables reading — use this when an upstream transform
            (``LoadImaged``) already loaded the image.
        """
        if isinstance(processor, str):
            from transformers import AutoImageProcessor

            processor = AutoImageProcessor.from_pretrained(processor)

        read = _resolve_reader(reader)

        def _run(x: Any):
            if read is not None and isinstance(x, (str, os.PathLike)):
                x = read(x)
            return processor(images=x, return_tensors="pt")["pixel_values"][0]

        return cls(keys, preprocess=_run)


class RadiologyEncoderTransform(_RadiologyTransformBase):
    """Fluent transform builder that uses an encoder's canonical preprocessor
    as the preprocessing step, then layers RadHarmony augmentations on top.

    Parallels :class:`RadiologyTransform2D` / :class:`RadiologyTransform3D` but
    replaces the ``load → normalise → resize → pad`` preprocessing with an
    :class:`EncoderPreprocessTransform` (typically built via
    :meth:`EncoderPreprocessTransform.from_huggingface` or
    :meth:`EncoderPreprocessTransform.from_custom`). Use this when pairing a
    RadHarmony dataset with a pretrained foundation model — the encoder's
    own processor handles image loading, resize, and normalization.

    The ``with_*`` augmentation hooks and the postprocessing
    (``ToTensorD`` + ``SelectItemsD`` over ``output_keys``) match
    :class:`RadiologyTransform2D`. Bbox-as-mask is not supported here —
    the encoder's processor resizes the image internally and cannot keep a
    bbox mask aligned.

    Parameters
    ----------
    preprocess : EncoderPreprocessTransform or callable
        Dict-transform that reads ``sample[key]`` and writes a
        ``Tensor[C, H, W]`` back to it.
    output_keys : set[str], optional
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation tasks — requires ``mask_preprocess`` to be set.
    mask_preprocess : callable, optional
        ``(path | image-like) -> Tensor[C, H, W]`` applied to the ``"mask"``
        key when ``"mask"`` is in *output_keys*. Must replicate the same
        **spatial** pipeline as *preprocess* (resize + crop to the same
        output resolution) but with nearest-neighbour interpolation and
        **no intensity normalisation**. Use the backbone factory (e.g.
        ``make_raddino(output_keys={"img","mask"})``) which wires this up
        automatically.
    dtype : torch.dtype, default ``torch.bfloat16``
        Dtype the final ``ToTensorD`` casts the output tensors to (``img``,
        ``cls``, ``mask``, ``bbox``). The default matches the project-wide
        convention assumed by ``BaseClsEvaluator._autocast_cm`` /
        ``BaseSegEvaluator._autocast_cm``, which bridge bf16 dataset output
        to fp32 encoder weights via autocast. Override to ``torch.float32``
        when pairing with an encoder/recipe that cannot autocast.

    Example
    -------
    ::

        from radharmony.evaluator import (
            EncoderPreprocessTransform,
            RadiologyEncoderTransform,
        )

        pp = EncoderPreprocessTransform.from_huggingface(
            keys=["img"], processor="microsoft/rad-dino",
        )
        # Classification
        transform = (
            RadiologyEncoderTransform(preprocess=pp, output_keys={"img", "cls"})
            .with_flip(spatial_axis=1)
            .get_transform()
        )
        # Segmentation — use make_raddino(output_keys={"img","mask"}) instead of
        # wiring mask_preprocess manually; shown here for illustration only.
        transform = (
            RadiologyEncoderTransform(
                preprocess=pp,
                output_keys={"img", "mask"},
                mask_preprocess=my_mask_preprocess_fn,
            )
            .get_transform()
        )
    """

    def __init__(
        self,
        preprocess: Callable[[dict], dict],
        output_keys: set[str] | None = None,
        mask_preprocess: Callable | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__(img_size=0, output_keys=output_keys, pad=False)
        self.preprocess_transform = preprocess
        self._mask_preprocess = mask_preprocess
        self.dtype = dtype

    def _build_preprocessing(self) -> list:
        steps: list = []
        if "mask" in self.output_keys and self._mask_preprocess is not None:
            steps.append(
                EncoderPreprocessTransform.from_custom(
                    keys=["mask"], preprocess=self._mask_preprocess
                )
            )
        steps.append(self.preprocess_transform)
        return steps

    def _build_postprocessing(self) -> list:
        tensor_keys = [
            k for k in ["img", "cls", "mask", "bbox"] if k in self.output_keys
        ]
        select_keys = [
            k
            for k in ["img", "cls", "mask", "report", "bbox", "bbox_labels"]
            if k in self.output_keys
        ]
        return [
            mt.ToTensorD(keys=tensor_keys, dtype=self.dtype, track_meta=False),
            mt.SelectItemsD(keys=select_keys),
        ]
