"""Evaluator-side MONAI transforms.

This module ships two pieces that plug a foundation model's canonical
preprocessing into a RadHarmony dataset pipeline:

* :class:`EncoderPreprocessTransform`: a thin ``MapTransform`` that calls
  ``preprocess(sample[key])`` and nothing else. The caller's ``preprocess``
  owns the full contract (reading, resizing, normalizing). Use
  :meth:`~EncoderPreprocessTransform.from_huggingface` for HF processors
  and :meth:`~EncoderPreprocessTransform.from_custom` for anything else.
* :class:`RadiologyEncoderTransform`: a fluent builder like
  :class:`radharmony.dataset.transforms.RadiologyTransform2D`. It uses
  :class:`EncoderPreprocessTransform` as the preprocessing step, so you can
  add augmentations with the ``.with_*`` hooks.

Reading images inside the transform
-----------------------------------

``EncoderPreprocessTransform.from_huggingface`` / ``.from_custom`` accept
a ``reader`` kwarg so the encoder's native file reader runs *here*,
bypassing any MONAI ``LoadImaged`` upstream.
This matters because each encoder was pretrained with one reader.
HF, timm and OpenCLIP models use ``PIL.Image.open``. Volumetric CXR models
use a DICOM or NIfTI reader. A different reader silently changes the pixel
statistics.

- ``reader="pil"``: ``PIL.Image.open(path).convert("RGB")``. Matches how
  HF ViT checkpoints were pretrained. PIL cannot open DICOM.
- ``reader="monai"``: ``mt.LoadImage(ensure_channel_first=True)``. A palette
  or RGBA PNG reads as the RGB image it shows, not as its raw palette index
  or with a 4th channel. An LA PNG reads as gray (D-13). Then
  it removes a trailing frame dim (a 2D DICOM loads as ``(C, W, H, 1)``)
  and transposes ``(C, W, H)`` to ``(C, H, W)``. For a DICOM file, it runs
  the 2 DICOM steps of the dataset load (HighBit, and the VOI window of a
  MONOCHROME image, D-13).
  Last, it repeats 1 channel to 3, because HF vision processors (RAD-DINO,
  ViT, ...) normalize 3 channels. Useful for DICOM / NIfTI.
- ``reader=<callable>``: your own ``(path) -> image-like``.
- ``reader=None``: the upstream dataset already loaded the image. Skip
  the read step and pass the value straight to the processor.

The ``"pil"`` and ``"monai"`` readers give the 0 to 255 range that an HF
processor expects (D-27 in ``docs/wiki/design.md``). An 8-bit image stays as
it is. The reader scales any other image (for example a 12-bit or 16-bit
image) to 0 to 255 with the minimum and maximum of that image. A reader never
clips. The DICOM window of ``reader="monai"`` is the display step of the file
(D-13), not a clip by the reader. For example, a 16-bit PNG with values 1,000 to 3,000 gives 0 to 255,
not 255 for every pixel.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Callable

import monai.transforms as mt
import numpy as np
import pydicom
import torch
from monai.config import KeysCollection
from monai.data import PILReader
from PIL import Image

from radharmony.dataset.base import _apply_voi_lut, _fix_missing_highbit
from radharmony.dataset.transforms import _BboxToTensorD, _ParseBboxD, _RadiologyTransformBase
from radharmony.utils.dicom import PHOTOMETRIC_TAG, is_dicom


def _scale_to_255(x):
    """Scale an array to 0 to 255 with its minimum and maximum (D-27). A uniform array gives 0."""
    lo, hi = x.min(), x.max()
    return (x - lo) * (255.0 / (hi - lo)) if hi > lo else x * 0


def _read_pil_8bit(path) -> Image.Image:
    """Open an image with PIL as 8-bit RGB. Scale a 16-bit or 32-bit image. Do not clip it."""
    img = Image.open(path)
    # Modes "I;16", "I;16B", "I" and "F" hold more than 8 bits. convert("RGB") clips them at 255.
    if img.mode.startswith("I") or img.mode == "F":
        arr = _scale_to_255(np.asarray(img, dtype=np.float64))
        img = Image.fromarray(np.round(arr).astype(np.uint8))
    return img.convert("RGB")


def _float_for_dicom_steps(x):
    """Give a DICOM image as float32 when the window or HighBit step writes new values (D-13).

    The 2 steps write their values back into the tensor. An integer tensor
    would wrap them, for example a 12-bit VOI LUT in an 8-bit image. An image
    that no step changes keeps its type, so an 8-bit file stays as it is (D-27).
    The window step skips an image that is not MONOCHROME, for example RGB (V-P-21).
    """
    path = str(x.meta.get("filename_or_obj", ""))
    if not is_dicom(path, PHOTOMETRIC_TAG in x.meta):
        return x
    ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    mono = ds.get("PhotometricInterpretation", "") in ("MONOCHROME1", "MONOCHROME2")
    if (mono and ("WindowCenter" in ds or "VOILUTSequence" in ds)) or "HighBit" not in ds:
        return x.float()
    return x


def _drop_palette_and_alpha(img: Image.Image) -> Image.Image:
    """Give the colors that a palette or alpha image shows, not its raw index or alpha (D-13).

    A palette (``P``, ``PA``) or ``RGBA`` image becomes RGB, and ``LA`` becomes gray.
    Other modes stay. Example: a Shenzhen PNG whose palette index 0 is white
    reads as white, not as 0.
    """
    if img.mode in ("P", "PA", "RGBA"):
        return img.convert("RGB")
    if img.mode == "LA":
        return img.convert("L")
    return img


class _LoadImageAsShown(mt.LoadImage):
    """``mt.LoadImage`` that reads a palette or alpha PNG as the image it shows (D-13).

    Each ``PILReader`` with no converter gets ``_drop_palette_and_alpha``, and it
    keeps its other options. MONAI still picks the reader by file suffix. PNG,
    JPEG and BMP files go to this reader. Other files go to the same reader as
    before. The arguments go to ``mt.LoadImage``. The class name is in the cache
    key (D-15), so a ``cache_dir`` cache from the old reader is not used. An
    ``embedding_cache`` file from the old reader raises ``ValueError`` (D-41).
    """

    def __init__(self, *args, **kwargs):
        # Not mt.LoadImage(reader=...): it tries every reader in turn, and NumpyReader
        # unpickles a .dcm file that holds a pickle.
        super().__init__(*args, **kwargs)
        for r in self.readers:
            if isinstance(r, PILReader) and r.converter is None:
                r.converter = _drop_palette_and_alpha


def _monai_to_255(x):
    """Give a MONAI-loaded image as float32 in 0 to 255. An 8-bit image stays as it is."""
    if x.dtype == torch.uint8:
        return x.float()
    return _scale_to_255(x.float())


def _display_255(x):
    """Give a MONAI-loaded 2-D image as the dataset load shows it, in 0 to 255 (D-13, D-27).

    For a DICOM file, it runs the HighBit fix, and the VOI window of a MONOCHROME image. An 8-bit image
    stays as it is. Any other image is scaled with its minimum and maximum. These
    are the steps of ``reader="monai"`` after the transpose. Input and output are
    ``(C, H, W)``. Load with ``dtype=None``, so that an 8-bit image keeps its type.
    """
    return _monai_to_255(_apply_voi_lut(_fix_missing_highbit(_float_for_dicom_steps(x))))


def _resolve_reader(reader: Any) -> Callable[[str], Any] | None:
    """Resolve a ``reader=`` keyword into a concrete ``(path) -> image`` callable."""
    if reader is None:
        return None
    if reader == "pil":
        return _read_pil_8bit
    if reader == "monai":
        # A 2D DICOM loads as (C, W, H, 1): squeeze the trailing frame dim.
        # A 2D NIfTI or PNG loads as (C, W, H): no squeeze needed.
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

        # dtype=None keeps the file's own dtype, so that _monai_to_255 can find an 8-bit image.
        # The 2 DICOM steps of the dataset load run before the scale to 0 to 255 (D-13).
        loader = mt.Compose(
            [
                _LoadImageAsShown(image_only=True, ensure_channel_first=True, dtype=None),
                mt.Lambda(_squeeze_frame_dim),
                mt.Transpose(indices=[0, 2, 1]),
                mt.Lambda(_float_for_dicom_steps),
                mt.Lambda(_fix_missing_highbit),
                mt.Lambda(_apply_voi_lut),
                mt.Lambda(_monai_to_255),
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
    processor : HF processor, optional
        The HF processor that ``preprocess`` calls, if any. The box warning of
        :class:`RadiologyEncoderTransform` reads its crop and pad settings (D-28).
    allow_missing_keys : bool
        Passed through to :class:`monai.transforms.MapTransform`.
    """

    def __init__(
        self,
        keys: KeysCollection,
        preprocess: Callable[[Any], Any],
        *,
        processor: Any = None,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys=allow_missing_keys)
        self.preprocess = preprocess
        # Private, so the cache description of the transform does not change (D-15).
        self._processor = processor

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
        *,
        processor: Any = None,
    ) -> "EncoderPreprocessTransform":
        """Wrap an arbitrary ``preprocess`` callable.

        ``preprocess`` owns the full contract: reading (if the input is a
        path), resizing, normalizing, and returning a ``Tensor[C, H, W]``.
        Same as the base constructor. It is a classmethod so the
        non-HuggingFace path is easy to find next to
        :meth:`from_huggingface`.

        Parameters
        ----------
        keys : KeysCollection
        preprocess : callable
            ``(path | image-like) -> torch.Tensor[C, H, W]``.
        processor : HF processor, optional
            The HF processor that ``preprocess`` calls, if any. Without it,
            a ``bbox`` output always gives the D-28 warning.
        """
        return cls(keys, preprocess=preprocess, processor=processor)

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
            ``AutoImageProcessor.from_pretrained(..., size=...)``. The
            schema (``shortest_edge``, ``height``/``width`` or
            ``longest_edge``) depends on the model family.
        reader : {'pil', 'monai', callable, None}, default 'pil'
            How to load the image when ``sample[key]`` is a path string.
            ``None`` turns reading off. Use it when an upstream transform
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

        return cls(keys, preprocess=_run, processor=processor)


def _keeps_box_fractions(preprocess) -> bool:
    """Return True when the encoder processor only resizes, so a box in fractions stays correct (D-28).

    A crop or a pad changes the fractions. ConvNeXt crops with ``crop_pct``.
    A custom ``preprocess`` with no processor has unknown steps, so it gives False.
    """
    proc = getattr(preprocess, "_processor", None)
    proc = getattr(proc, "image_processor", proc)  # an AutoProcessor holds the image processor
    if proc is None:
        return False
    crops = getattr(proc, "do_center_crop", False) or getattr(proc, "crop_pct", None) is not None
    return not crops and not getattr(proc, "do_pad", False)


class _DarkestToMinusOneD:
    """Shift each image channel so that its darkest value is -1, or undo the shift (D-37).

    The base warp step fills its empty border with -1. With this shift before
    the warp and the undo after it, the border gets the darkest value of each
    channel. The shift is kept in the sample under ``_darkest`` until the undo.
    """

    def __init__(self, undo: bool = False):
        self.undo = undo

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        img = d["img"]
        if self.undo:
            d["img"] = img + d.pop("_darkest") + 1
        else:
            darkest = img.reshape(img.shape[0], -1).amin(dim=1).reshape(-1, *([1] * (img.ndim - 1)))
            d["_darkest"] = darkest
            d["img"] = img - darkest - 1
        return d


class RadiologyEncoderTransform(_RadiologyTransformBase):
    """Fluent transform builder that uses an encoder's canonical preprocessor
    as the preprocessing step, then layers RadHarmony augmentations on top.

    Parallels :class:`RadiologyTransform2D` / :class:`RadiologyTransform3D` but
    replaces the ``load → normalise → resize → pad`` preprocessing with an
    :class:`EncoderPreprocessTransform` (typically built via
    :meth:`EncoderPreprocessTransform.from_huggingface` or
    :meth:`EncoderPreprocessTransform.from_custom`). Use this when pairing a
    RadHarmony dataset with a pretrained foundation model. The encoder's
    own processor does image loading, resize, and normalization.

    The ``with_*`` augmentation hooks and the postprocessing
    (``ToTensorD`` + ``SelectItemsD`` over ``output_keys``) match
    :class:`RadiologyTransform2D`. Bbox-as-mask is not supported here,
    because the encoder's processor resizes the image internally and cannot
    keep a bbox mask aligned. The ``bbox`` key does not follow the crop or pad
    of the processor either. So the transform warns when ``bbox`` is in
    ``output_keys`` and the processor does more than resize (D-28).

    Parameters
    ----------
    preprocess : EncoderPreprocessTransform or callable
        Dict-transform that reads ``sample[key]`` and writes a
        ``Tensor[C, H, W]`` back to it.
    output_keys : set[str], optional
        Keys to keep in each sample dict. Defaults to ``{"img", "cls"}``.
        Pass ``{"img", "mask"}`` or ``{"img", "cls", "mask"}`` for
        segmentation tasks. This needs ``mask_preprocess``.
    mask_preprocess : callable, optional
        ``(path | image-like) -> Tensor[C, H, W]`` applied to the ``"mask"``
        key when ``"mask"`` is in *output_keys*. Must replicate the same
        **spatial** pipeline as *preprocess* (resize + crop to the same
        output resolution) but with nearest-neighbour interpolation and
        **no intensity normalization**. The backbone factory (for example
        ``make_raddino(output_keys={"img","mask"})``) sets this up
        for you.
    dtype : torch.dtype, default ``torch.bfloat16``
        Dtype the final ``ToTensorD`` casts the image (``img``) to. The
        labels (``cls``, ``reg``, ``mask``) are always float32. So a mask
        id 257 stays 257 (V-B-16). ``bbox`` is always float32 (D-28).
        The default matches the project-wide
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
        # Segmentation: use make_raddino(output_keys={"img","mask"}) instead of
        # wiring mask_preprocess by hand. This is only an example.
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
        if "mask" in self.output_keys and mask_preprocess is None:
            raise ValueError(
                "output_keys has 'mask', so mask_preprocess is required. Without it, "
                "'mask' stays a file path. Use a backbone factory, for example "
                "make_raddino(output_keys={'img', 'mask'}), or pass mask_preprocess."
            )
        if "bbox" in self.output_keys and not _keeps_box_fractions(preprocess):
            warnings.warn(
                "The box does not follow the crop or pad of the encoder processor, so 'bbox' can be "
                "wrong in the sample (D-28). A processor that only resizes keeps it correct. A custom "
                "preprocess with no processor always warns, because its steps are unknown.",
                UserWarning,
                stacklevel=2,
            )
        self.preprocess_transform = preprocess
        self._mask_preprocess = mask_preprocess
        self.dtype = dtype

    def _append_warp(self, warp_cls, **kwargs) -> None:
        """Append the base warp, so that its empty border gets the darkest value of each channel (D-37).

        The encoder has already normalized the image, so the -1 of the base
        step is a gray there. Example: for RAD-DINO, black is -2.055.
        """
        super()._append_warp(warp_cls, **kwargs)
        warp = self._aug.pop()
        self._aug.append(mt.Compose([_DarkestToMinusOneD(), warp, _DarkestToMinusOneD(undo=True)]))

    def _build_preprocessing(self) -> list:
        # The box, its labels and its findings become plain lists first, as in
        # RadiologyTransform2D (D-28).
        steps: list = [_ParseBboxD(2)] if "bbox" in self.output_keys else []
        if "mask" in self.output_keys:
            steps.append(
                EncoderPreprocessTransform.from_custom(
                    keys=["mask"], preprocess=self._mask_preprocess
                )
            )
        steps.append(self.preprocess_transform)
        return steps

    def _build_postprocessing(self) -> list:
        # The box gets its own step: a float32 (N, 4) tensor, also for N = 0 (D-28).
        # dtype is for the image only. The labels stay float32, because bfloat16
        # rounds a mask id 257 to 256 and a label 37.3 to 37.25 (V-B-16).
        label_keys = [k for k in ["cls", "reg", "mask"] if k in self.output_keys]
        select_keys = [
            k
            for k in ["img", "cls", "reg", "mask", "report", "bbox", "bbox_labels", "bbox_findings"]
            if k in self.output_keys
        ]
        box_step = [_BboxToTensorD(2)] if "bbox" in self.output_keys else []
        steps = box_step
        if label_keys:
            steps = steps + [mt.ToTensorD(keys=label_keys, dtype=torch.float32, track_meta=False)]
        if "img" in self.output_keys:
            steps = steps + [mt.ToTensorD(keys=["img"], dtype=self.dtype, track_meta=False)]
        return steps + [mt.SelectItemsD(keys=select_keys)]
