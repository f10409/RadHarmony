"""User-facing transform builder for radiological image pipelines.

Wraps the base 2-D / 3-D preprocessing steps from :mod:`radharmony.dataset.base`
and exposes a fluent API for layering common augmentations on top.

Typical usage::

    transform = (
        RadiologyTransform2D(img_size=224, output_keys={"img", "cls"})
        .with_flip()
        .with_affine(translate_range=(10, 10), rotate_range=(0.17,), scale_range=(0.1, 0.1))
        .with_intensity_jitter(shift=0.1, scale=0.1)
        .get_transform()
    )
"""

from __future__ import annotations

import ast
import json
import math

import pandas as pd
import torch
import monai as mn
from monai.utils import ensure_tuple

from radharmony.dataset.base import (
    _apply_voi_lut,
    _fix_missing_highbit,
)


# ---------------------------------------------------------------------------
# Box helpers (D-28)
# ---------------------------------------------------------------------------
# The box in a sample must match the image in that sample. Resize, pad, flip
# and transpose move the box with exact math on its corners. A resize keeps the
# fractions as they are, because it stretches the whole image. Only the affine
# and elastic warps use a mask round trip. Each box gets its own mask channel.
# The warp moves the masks with the image. Then the boxes are read back from
# the masks.

_BOX_LIST_KEYS = ("bbox_labels", "bbox_findings")


def _parse_list(raw) -> list:
    """Return a per-box value (boxes, labels or findings) as a list.

    A JSON string is parsed. A Python list text, as ``to_csv`` writes it (for
    example ``"['a', 'b']"``), is parsed with ``ast.literal_eval`` (V-D-33). A
    missing value (None, NaN, ``pd.NA``, "nan", "") gives [].
    """
    if raw is None or (not hasattr(raw, "__len__") and pd.isna(raw)):
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if text in ("", "nan", "None"):
            return []
        try:
            return list(json.loads(text))
        except ValueError as e:
            try:
                value = ast.literal_eval(text)  # reads only literals, never runs code
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                value = None
            if isinstance(value, (list, tuple)):
                return list(value)
            raise ValueError(
                f"cannot read the box, label or finding value {text[:80]!r} as JSON "
                f"or as a Python list: {e}"
            ) from e
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    return list(raw)


def _parse_boxes(raw, ndim: int) -> list[list[float]]:
    """Return the box value of a sample as a list of boxes.

    Args:
        raw: a JSON string, one flat box, a list of boxes, a tensor or an
            array. A missing value (None, NaN, "nan", "") means no box.
        ndim: number of spatial dimensions (2 or 3).

    Returns:
        A list of boxes. Each box is a list of ``2 * ndim`` floats.

    Raises:
        ValueError: a box does not have ``2 * ndim`` values, or a value is
            not finite.
    """
    boxes = _parse_list(raw)
    if boxes and not isinstance(boxes[0], (list, tuple)) and getattr(boxes[0], "ndim", 0) == 0:
        boxes = [boxes]  # one flat box, for example 6 numpy floats (RAD-ChestCT)
    out = [[float(v) for v in _parse_list(b)] for b in boxes]
    for b in out:
        if len(b) != 2 * ndim:
            raise ValueError(f"a {ndim}-D box needs {2 * ndim} values, got {b}")
        if not all(math.isfinite(v) for v in b):
            raise ValueError(f"a box value is not finite (D-07): {b}")
    return out


def _sample_boxes(d: dict) -> list[list[float]]:
    """Return the boxes of a sample dict as a list (see :func:`_parse_boxes`).

    The number of spatial dimensions comes from ``d["img"]``. So a box step
    also works in a pipeline with no :class:`_ParseBboxD`, for example a JSON
    string box in a custom pipeline.
    """
    return _parse_boxes(d.get("bbox"), d["img"].ndim - 1)


def _flip_box(box: list[float], axes) -> list[float]:
    """Return *box* flipped along each spatial axis in *axes*."""
    out = list(box)
    for a in axes:
        lo, hi = out[2 * a], out[2 * a + 1]
        out[2 * a], out[2 * a + 1] = 1.0 - hi, 1.0 - lo
    return out


def _keep_boxes(data: dict, keep: list[int]) -> None:
    """Keep only the per-box labels and findings at the indices in *keep*."""
    for key in _BOX_LIST_KEYS:
        if key in data:
            values = _parse_list(data[key])
            data[key] = [values[i] for i in keep if i < len(values)]


def _paint_box(channel: torch.Tensor, box: list[float]) -> None:
    """Fill one box into a ``(*spatial)`` mask channel (changed in place).

    Each edge goes to the nearest pixel edge, so a round trip moves an edge by
    0.5 pixel or less. Values outside [0, 1] are clipped. A box always covers
    at least 1 pixel on each axis.
    """
    index = []
    for i, n in enumerate(channel.shape):
        lo = min(max(box[2 * i], 0.0), 1.0)
        hi = min(max(box[2 * i + 1], 0.0), 1.0)
        start = min(math.floor(lo * n + 0.5), n - 1)
        stop = max(math.floor(hi * n + 0.5), start + 1)
        index.append(slice(start, stop))
    channel[tuple(index)] = 1


def _paint_mask(boxes: list[list[float]], spatial: tuple) -> torch.Tensor:
    """Return a float32 mask of shape ``(N, *spatial)`` with box ``c`` in channel ``c``."""
    mask = torch.zeros(len(boxes), *spatial, dtype=torch.float32)
    for c, box in enumerate(boxes):
        _paint_box(mask[c], box)
    return mask


def _channel_to_box(channel: torch.Tensor) -> list[float] | None:
    """Return the box of the nonzero pixels of one mask channel, or None."""
    nonzero = torch.nonzero(channel, as_tuple=False)
    if nonzero.numel() == 0:
        return None
    box: list[float] = []
    for i, n in enumerate(channel.shape):
        # The upper edge is the far side of the last pixel: (index + 1) / n.
        box += [nonzero[:, i].min().item() / n, (nonzero[:, i].max().item() + 1) / n]
    return box


class _ParseBboxD(mn.transforms.Transform):
    """Turn ``bbox``, ``bbox_labels`` and ``bbox_findings`` into plain lists.

    It runs first, so every later box step gets a list of boxes. No-op when
    ``"bbox"`` is absent.
    """

    def __init__(self, ndim: int):
        self.ndim = ndim

    def __call__(self, data: dict) -> dict:
        if "bbox" not in data:
            return data
        d = dict(data)
        d["bbox"] = _parse_boxes(d["bbox"], self.ndim)
        for key in _BOX_LIST_KEYS:
            if key in d:
                d[key] = _parse_list(d[key])
        return d


class _SpatialPadBboxD(mn.transforms.SpatialPadd):
    """``SpatialPadd`` that also moves the box with the pad of ``"img"``."""

    def __call__(self, data, lazy=None):
        shape = tuple(data["img"].shape[1:]) if "img" in data else None
        d = super().__call__(data, lazy)
        if shape is None or "bbox" not in d:
            return d
        width = self.padder.compute_pad_width(shape)[1:]  # skip the channel axis
        boxes = []
        for box in _sample_boxes(d):
            out = []
            for i, (n, (before, after)) in enumerate(zip(shape, width, strict=True)):
                total = n + before + after
                out += [(box[2 * i] * n + before) / total, (box[2 * i + 1] * n + before) / total]
            boxes.append(out)
        d["bbox"] = boxes
        return d


class _SpacingBboxD(mn.transforms.SpacingD):
    """``SpacingD`` that also moves the box with the resample of ``"img"`` (D-28).

    The output of a resample covers a different area than the input, because
    MONAI keeps the first voxel center and rounds the output size. So the box
    fractions change. Each box corner goes through the voxel map of the two
    affines. The new box is clipped to [0, 1]. A box on the strip that the
    resample cuts off has no area then. The step drops it with its label and
    finding (D-07).

    Args:
        box_axes: the image axis of each box pair at this step. ``(2, 1, 0)``
            when a ``Transposed([0, 3, 2, 1])`` comes after this step.
        Other arguments go to ``SpacingD``.
    """

    def __init__(self, *args, box_axes=(0, 1, 2), **kwargs):
        super().__init__(*args, **kwargs)
        self.box_axes = tuple(box_axes)

    def __call__(self, data, lazy=None):
        img = data.get("img")
        if "bbox" not in data or not hasattr(img, "affine"):
            return super().__call__(data, lazy)
        import numpy as np

        shape, a_in = img.shape[1:], np.asarray(img.affine, dtype=np.float64)
        d = super().__call__(data, lazy)
        out = d["img"]
        shape_out = out.shape[1:]
        m = np.linalg.solve(np.asarray(out.affine, dtype=np.float64), a_in)  # input voxel -> output voxel
        boxes, keep = [], []
        for c, box in enumerate(_sample_boxes(d)):
            # Box edges as voxel-center coordinates (edge - 0.5), in image axis order.
            lo, hi = [0.0] * 3, [0.0] * 3
            for k, ax in enumerate(self.box_axes):
                lo[ax], hi[ax] = box[2 * k] * shape[ax] - 0.5, box[2 * k + 1] * shape[ax] - 0.5
            corners = np.array(np.meshgrid(*zip(lo, hi, strict=True), indexing="ij")).reshape(3, -1)
            mapped = m[:3, :3] @ corners + m[:3, 3:]
            new = []
            for ax in self.box_axes:
                n = shape_out[ax]
                new += [min(max((mapped[ax].min() + 0.5) / n, 0.0), 1.0),
                        min(max((mapped[ax].max() + 0.5) / n, 0.0), 1.0)]
            if all(new[2 * k] < new[2 * k + 1] for k in range(3)):
                boxes.append(new)
                keep.append(c)
        d["bbox"] = boxes
        _keep_boxes(d, keep)
        return d


class _RandFlipBboxD(mn.transforms.RandFlipd):
    """``RandFlipd`` that also flips the box when it flips the image."""

    def __call__(self, data, lazy=None):
        d = super().__call__(data, lazy)
        if self._do_transform and "bbox" in d and "img" in self.keys:
            ndim = d["img"].ndim - 1
            axis = self.flipper.spatial_axis
            axes = range(ndim) if axis is None else [a % ndim for a in ensure_tuple(axis)]
            d["bbox"] = [_flip_box(b, axes) for b in _sample_boxes(d)]
        return d


class _TransposeBboxD(mn.transforms.Transposed):
    """``Transposed`` that also swaps the box axes when it transposes the image."""

    def __call__(self, data):
        d = super().__call__(data)
        if "bbox" in d and "img" in self.keys:
            indices = self.transform.indices
            if indices is None or indices[0] != 0:
                raise ValueError(
                    "with_transpose() must keep the channel axis first to move the box"
                )
            axes = [i - 1 for i in indices[1:]]  # new spatial axis k <- old axis axes[k]
            d["bbox"] = [
                [v for a in axes for v in (b[2 * a], b[2 * a + 1])] for b in _sample_boxes(d)
            ]
        return d


class _BboxToMask(mn.transforms.Transform):
    """Paint each box into its own channel of ``"bbox_mask"``.

    The mask has the spatial size of ``"img"`` and one channel for each box.
    It runs just before an affine or elastic warp. An image with no box gets
    one empty channel, so the warp always finds ``"bbox_mask"``. No-op when
    ``"bbox"`` is absent.
    """

    def __call__(self, data: dict) -> dict:
        if "bbox" not in data:
            return data
        d = dict(data)
        d["bbox"] = boxes = _sample_boxes(d)
        spatial = tuple(d["img"].shape[1:])
        d["bbox_mask"] = (_paint_mask(boxes, spatial) if boxes
                          else torch.zeros(1, *spatial, dtype=torch.float32))
        return d


class _MaskToBbox(mn.transforms.Transform):
    """Read the boxes back from ``"bbox_mask"`` after an affine or elastic warp.

    A channel that the warp did not change keeps its exact box. A channel that
    is empty after the warp (the box left the image) drops its box, label and
    finding. When no box is left, ``"bbox"`` is an empty list. It removes
    ``"bbox_mask"``.
    """

    def __call__(self, data: dict) -> dict:
        if "bbox_mask" not in data:
            return data
        d = dict(data)
        mask = d.pop("bbox_mask")
        boxes, keep = [], []
        for c, box in enumerate(d["bbox"]):
            painted = torch.zeros(mask.shape[1:], dtype=torch.float32)
            _paint_box(painted, box)
            channel = torch.as_tensor(mask[c]) > 0.5
            if torch.equal(channel, painted > 0.5):
                new = box  # the warp did not run (or did not move this box)
            else:
                new = _channel_to_box(channel)
            if new is not None:
                boxes.append(new)
                keep.append(c)
        d["bbox"] = boxes
        _keep_boxes(d, keep)
        return d


class _BboxToTensorD(mn.transforms.Transform):
    """Give the boxes as a float32 tensor of shape ``(N, 2 * ndim)``.

    ``N`` is 0 when the image has no box. With ``paint_mask=True`` it also gives
    ``"bbox_mask"``: one channel for each box, on the grid of ``"img"``.
    """

    def __init__(self, ndim: int, paint_mask: bool = False, mask_dtype=torch.float32):
        self.ndim = ndim
        self.paint_mask = paint_mask
        self.mask_dtype = mask_dtype

    def __call__(self, data: dict) -> dict:
        if "bbox" not in data:
            return data
        d = dict(data)
        boxes = _parse_boxes(d["bbox"], self.ndim)
        d["bbox"] = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 2 * self.ndim)
        if self.paint_mask:
            d["bbox_mask"] = _paint_mask(boxes, tuple(d["img"].shape[1:])).to(self.mask_dtype)
        return d


# ---------------------------------------------------------------------------
# Bbox orientation helper
# ---------------------------------------------------------------------------

_WORLD_AXIS = {'R': 0, 'L': 0, 'A': 1, 'P': 1, 'S': 2, 'I': 2}
_AXIS_SIGN  = {'R': 1, 'L': -1, 'A': 1, 'P': -1, 'S': 1, 'I': -1}


def _perm_flip_between_codes(src_code: str, tgt_code: str):
    """Return (sigma, flip_signs) mapping src orientation axes to tgt orientation axes.

    sigma[i]      – source axis index feeding target axis i
    flip_signs[i] – +1.0 (same direction) or -1.0 (needs flip)
    """
    sigma, flip_signs = [], []
    for tc in tgt_code.upper():
        tw, ts = _WORLD_AXIS[tc], _AXIS_SIGN[tc]
        for j, sc in enumerate(src_code.upper()):
            if _WORLD_AXIS[sc] == tw:
                sigma.append(j)
                flip_signs.append(1.0 if _AXIS_SIGN[sc] == ts else -1.0)
                break
    return sigma, flip_signs


class _ReorientBbox(mn.transforms.Transform):  # a MONAI Transform, so the cache goes past it (D-15)
    """Permute bbox coord pairs to stay aligned after OrientationD(axcodes)+TransposeD([0,3,2,1]).

    Must run AFTER EnsureChannelFirstD (affine populated) and BEFORE OrientationD.
    Uses SimpleITK to read the source orientation code from the image direction
    cosines, so it does not need nibabel. Does nothing when ``"bbox"`` is absent.

    Args:
        axcodes: Target orientation string (for example ``"RAS"``, ``"LPS"``, ``"IPL"``).
                 Must match the sibling ``OrientationD`` exactly.

    Math:
        The harmonizer encodes ``bbox`` in **post-Transpose source MONAI dim
        order** (pair ``k`` represents source MONAI dim ``2-k``).  After
        ``_SITKOrientD(axcodes)`` + ``TransposeD([0,3,2,1])``, final MONAI dim
        ``i`` carries the direction of target axcodes letter ``i``.
        ``_perm_flip_between_codes(src_code, axcodes)`` returns ``sigma[i]`` =
        source axcodes index whose direction matches target axis ``i``, plus
        ``flip_signs[i]`` (±1) for the relative orientation.  So new pair
        ``i`` ← old pair ``sigma[i]``, with ``[lo, hi] → [1-hi, 1-lo]`` when
        ``flip_signs[i] < 0``.

        (``src_code`` comes from the helper below. The helper builds direction
        cosines for the SITK image that ``_SITKOrientD`` uses. That string is
        the *reverse* of the MONAI-standard axcodes for the affine. This is
        why the formula uses ``sigma`` directly and not ``2 - sigma[2-i]``.)
    """

    def __init__(self, axcodes: str = "RAS"):
        self.axcodes = axcodes

    def _src_orientation_code(self, ras_affine) -> str:
        """Return the 3-letter ITK orientation code using SimpleITK."""
        import numpy as np
        import SimpleITK as sitk

        m = np.asarray(ras_affine, dtype=float)[:3, :3]
        norms = np.linalg.norm(m, axis=0)
        norms[norms < 1e-10] = 1.0
        d_ras = m / norms
        # MONAI affine is in RAS; SimpleITK uses LPS.
        # Negate rows 0 (R→L) and 1 (A→P) to convert direction to LPS.
        d_lps = d_ras.copy()
        d_lps[:2, :] *= -1
        # D_sitk[:,j] = unit direction of ITK axis j = d_lps[:,2-j]  (MONAI dim j = ITK axis 2-j)
        D_sitk = np.column_stack([d_lps[:, 2], d_lps[:, 1], d_lps[:, 0]])
        direction = D_sitk.flatten(order="C").tolist()
        return sitk.DICOMOrientImageFilter().GetOrientationFromDirectionCosines(direction)

    def __call__(self, data: dict) -> dict:
        if "bbox" not in data:
            return data
        img = data.get("img")
        if img is None or not hasattr(img, "affine"):
            return data

        # No try/except: an error must show. Else the image turns and the box keeps its old axes (R-code-18).
        affine = img.affine.numpy() if hasattr(img.affine, "numpy") else img.affine
        src_code = self._src_orientation_code(affine)
        sigma, flip_signs = _perm_flip_between_codes(src_code, self.axcodes)

        # Final MONAI dim i ← harmonizer pair sigma[i], with the flip from
        # flip_signs[i] (see class docstring for the derivation).
        coord_perm = list(sigma)
        flips      = list(flip_signs)

        def _xfm(b):
            out = []
            for i in range(3):
                j = coord_perm[i]
                lo, hi = float(b[j * 2]), float(b[j * 2 + 1])
                if flips[i] < 0:
                    lo, hi = 1.0 - hi, 1.0 - lo
                out += [lo, hi]
            return out

        # A list of boxes, also for one flat box. No box gives [] (D-28, V-D-41).
        data["bbox"] = [_xfm(b) for b in _sample_boxes(data)]

        return data


class _SITKOrientD(mn.transforms.Transform):  # a MONAI Transform, so the cache goes past it (D-15)
    """Reorient 3-D MetaTensors to a canonical orientation using SimpleITK.

    Drop-in replacement for ``mn.transforms.OrientationD`` that uses
    SimpleITK's ``DICOMOrientImageFilter`` instead of nibabel.  Each key
    must be a MetaTensor with a populated RAS affine (produced by ITKReader +
    EnsureChannelFirstD).  Keys without a MetaTensor affine are skipped.

    Axis-reversal convention (MONAI spatial dim j = ITK axis 2-j):
    - ``tensor[0].numpy()`` shape ``(D, H, W)`` is passed directly to
      ``sitk.GetImageFromArray``, which treats it as ``(z, y, x)``. This is correct.
    - After reorientation, ``sitk.GetArrayFromImage`` returns ``(sz, sy, sx)``,
      which maps back to MONAI ``(dim0, dim1, dim2)``. This is also correct.

    Args:
        keys: Data dict keys to reorient.
        axcodes: Target 3-letter orientation string (for example ``"RAS"``, ``"LPS"``).
    """

    def __init__(self, keys: list[str], axcodes: str = "RAS"):
        self.keys = list(keys)
        self.axcodes = axcodes.upper()

    def _reorient_one(self, tensor, sitk_filter):
        import numpy as np
        import SimpleITK as sitk
        from monai.data import MetaTensor

        affine = tensor.affine
        if hasattr(affine, "numpy"):
            affine = affine.numpy()
        affine = np.asarray(affine, dtype=np.float64)

        m = affine[:3, :3]
        spacing = np.linalg.norm(m, axis=0)
        spacing = np.where(spacing < 1e-10, 1.0, spacing)
        d_ras = m / spacing
        d_lps = d_ras.copy()
        d_lps[:2, :] *= -1  # RAS→LPS: negate R(row 0) and A(row 1)

        # ITK axis j = MONAI dim (2-j), so D_sitk[:,j] = d_lps[:,2-j]
        D_sitk = np.column_stack([d_lps[:, 2], d_lps[:, 1], d_lps[:, 0]])
        sitk_spacing = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
        origin_lps = affine[:3, 3].copy()
        origin_lps[:2] *= -1  # RAS→LPS origin (negate R,A components)

        arr = tensor[0].to(torch.float32).contiguous().numpy()  # (D, H, W)
        sitk_img = sitk.GetImageFromArray(arr)  # (z,y,x)→size=(W,H,D)
        sitk_img.SetSpacing(sitk_spacing)
        sitk_img.SetDirection(D_sitk.flatten(order="C").tolist())
        sitk_img.SetOrigin(origin_lps.tolist())

        reoriented = sitk_filter.Execute(sitk_img)

        new_D_sitk = np.array(reoriented.GetDirection()).reshape(3, 3)
        new_spacing_sitk = np.array(reoriented.GetSpacing())  # (sx,sy,sz) ITK
        new_origin_lps = np.array(reoriented.GetOrigin())

        new_d_ras = new_D_sitk.copy()
        new_d_ras[:2, :] *= -1  # LPS→RAS: negate first two physical rows

        # MONAI affine col j = direction×spacing for MONAI dim j = ITK axis (2-j)
        new_affine_3x3 = np.column_stack([
            new_d_ras[:, 2] * new_spacing_sitk[2],  # MONAI dim 0 ← ITK axis 2
            new_d_ras[:, 1] * new_spacing_sitk[1],  # MONAI dim 1 ← ITK axis 1
            new_d_ras[:, 0] * new_spacing_sitk[0],  # MONAI dim 2 ← ITK axis 0
        ])
        new_origin_ras = new_origin_lps.copy()
        new_origin_ras[:2] *= -1  # LPS→RAS origin

        new_affine = np.eye(4, dtype=np.float64)
        new_affine[:3, :3] = new_affine_3x3
        new_affine[:3, 3] = new_origin_ras

        # GetArrayFromImage returns (size[2], size[1], size[0]) = (D, H, W) in MONAI order
        new_arr = sitk.GetArrayFromImage(reoriented)
        new_data = torch.from_numpy(new_arr.copy()).unsqueeze(0).to(tensor.dtype)
        return MetaTensor(
            new_data,
            affine=torch.as_tensor(new_affine, dtype=torch.float64),
        )

    def __call__(self, data: dict) -> dict:
        import SimpleITK as sitk

        f = sitk.DICOMOrientImageFilter()
        f.SetDesiredCoordinateOrientation(self.axcodes)

        for key in self.keys:
            if key not in data:
                continue
            tensor = data[key]
            if not hasattr(tensor, "affine"):
                continue
            data[key] = self._reorient_one(tensor, f)

        return data


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _binary_mask(x):
    """Map each mask value above 0 to 1 (D-14). Mask files often hold 0 and 255."""
    return (x > 0).to(x.dtype) if isinstance(x, torch.Tensor) else (x > 0).astype(x.dtype)


# Module-level functions, not lambdas: a "spawn" or "forkserver" DataLoader worker must pickle them.
def _drop_dicom_axis(x):
    """Make (C, H, W, 1) from a 2-D DICOM into (C, H, W). Other images pass through."""
    return x[..., 0] if x.ndim == 4 else x


def _shift_up(x):
    """Add 1, so that the warp border value 0 is -1 after ``_shift_down``."""
    return x + 1


def _shift_down(x):
    """Subtract 1 after the warp."""
    return x - 1


class _SetChannelsD(mn.transforms.Transform):
    """Give ``img`` the channel count of the dataset (D-42).

    It drops an alpha channel. With 1 channel, RGB becomes gray with the weights
    of PIL ``convert("L")``. With 3 channels, it repeats a gray image 3 times.

    Args:
        channels: 1 (gray modality) or 3 (color figures).
    """

    def __init__(self, channels: int = 1):
        if channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, not {channels!r} (D-42).")
        self.channels = channels  # public, so the cache key has it (D-15)

    def __call__(self, data: dict) -> dict:
        d = dict(data)
        x = d["img"]
        c = x.shape[0]
        if c in (2, 4):  # gray + alpha, RGB + alpha
            x, c = x[: c - 1], c - 1
        if c == 3 and self.channels == 1:
            same = (x[0] == x[1]).all() and (x[1] == x[2]).all()  # a gray image saved as RGB
            x = x[0:1] if same else 0.299 * x[0:1] + 0.587 * x[1:2] + 0.114 * x[2:3]
        elif c == 1 and self.channels == 3:
            x = x.repeat(3, 1, 1) if isinstance(x, torch.Tensor) else x.repeat(3, axis=0)
        elif c != self.channels:
            raise ValueError(f"An image with {c} channels cannot become {self.channels} channels (D-42).")
        d["img"] = x
        return d


def _base_load_2d(img_keys: list[str], base_transpose: bool = True, channels: int = 1) -> list:
    """Return the load and normalize steps shared by all 2-D pipelines.

    ``channels`` is the channel count of ``img`` (D-42): 1 for a gray modality, 3 for color figures.
    """
    t = [
        mn.transforms.LoadImageD(keys=img_keys, reader="ITKReader"),
        mn.transforms.EnsureChannelFirstD(keys=img_keys),
        # ITKReader loads 2-D DICOMs as (H, W, 1). After EnsureChannelFirst they are (C, H, W, 1).
        # Non-DICOM 2-D images (JPEG, PNG) load as (C, H, W).
        mn.transforms.Lambdad(keys=img_keys, func=_drop_dicom_axis),
    ]
    if base_transpose:
        # ITK axis correction: not needed when loading numpy arrays directly.
        t.append(mn.transforms.Transposed(keys=img_keys, indices=[0, 2, 1]))
    t += [
        mn.transforms.Lambdad(keys=["img"], func=_fix_missing_highbit),
        mn.transforms.Lambdad(keys=["img"], func=_apply_voi_lut),
        # No MONOCHROME1 inversion step: GDCM already inverts it when it reads the file (V-P-1).
        _SetChannelsD(channels),
        mn.transforms.ScaleIntensityRangePercentilesD(
            keys=["img"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True
        ),
    ]
    return t


def _base_load_3d(
    img_keys: list[str],
    hu_window: tuple[float, float] | None,
    base_transpose: bool = True,
    pixdim: tuple[float, float, float] | None = None,
    axcodes: str = "IPL",
) -> list:
    """Return the load and normalize steps shared by all 3-D pipelines.

    When ``pixdim`` is set, SpacingD resamples to that target voxel spacing
    (mm). It uses the MetaTensor affine from ITKReader: PixelSpacing in the
    plane, and the slice positions along z (not SliceThickness). This is
    important for anisotropic MRI. Without it, the longest-axis ResizeD
    flattens a 0.5 x 0.5 x 4 mm lumbar-spine series into about 7 slices.
    The box moves with the resample (``_SpacingBboxD``, D-28).

    ``axcodes`` sets the orientation target of ``_SITKOrientD`` and of the
    matching box step ``_ReorientBbox``. Both use the same value (default
    ``"IPL"``, D-12).
    """
    t = [
        mn.transforms.LoadImageD(keys=img_keys, reader="ITKReader"),
        mn.transforms.EnsureChannelFirstD(keys=img_keys),
    ]
    if base_transpose:
        # Reorient bbox coords BEFORE OrientationD so the coordinate pairs
        # stay aligned with the reoriented image (see _ReorientBbox docstring).
        t.append(_ReorientBbox(axcodes=axcodes))
        # Set the scanner axis order to one canonical orientation. Then all
        # studies load with the same axis meaning, for any patient position
        # or acquisition convention.
        t.append(_SITKOrientD(keys=img_keys, axcodes=axcodes))
    if pixdim is not None:
        # Bilinear for intensity, nearest for masks (keeps labels discrete).
        spacing_modes = tuple("bilinear" if k == "img" else "nearest" for k in img_keys)
        t.append(
            _SpacingBboxD(
                keys=img_keys,
                pixdim=pixdim,
                mode=spacing_modes,
                # Box pair k is image axis 2 - k until the transpose below.
                box_axes=(2, 1, 0) if base_transpose else (0, 1, 2),
            )
        )
    if base_transpose:
        # ITK axis correction: not needed when loading numpy arrays directly.
        #
        # ⚠ DO NOT REMOVE without reading docs/proposals/orientation_normalization.md
        # AND running tests/test_bbox_axis_alignment.py. This step is NOT only
        # cosmetic. The bbox harmonizers write bbox coordinates in the
        # POST-transpose (D, H, W) axis order (see commit d1d0535 for the lumbar
        # spine one). _ReorientBbox (above) changes the bbox coords first, so
        # they stay correct after this transpose.
        t.append(mn.transforms.Transposed(keys=img_keys, indices=[0, 3, 2, 1]))
    if hu_window is not None:
        t.append(
            mn.transforms.ScaleIntensityRangeD(
                keys=["img"],
                a_min=hu_window[0],
                a_max=hu_window[1],
                b_min=hu_window[0],
                b_max=hu_window[1],
                clip=True,
            )
        )
    t.append(
        mn.transforms.ScaleIntensityRangePercentilesD(
            keys=["img"], lower=0.5, upper=99.5, b_min=-1, b_max=1, clip=True
        )
    )
    return t


# ---------------------------------------------------------------------------
# Base builder
# ---------------------------------------------------------------------------


class _RadiologyTransformBase:
    """Internal base class shared by the 2-D and 3-D builders.

    Do not instantiate it directly. Use :class:`RadiologyTransform2D` or
    :class:`RadiologyTransform3D`.
    """

    _NDIM = 2  # number of spatial dimensions (the 3-D builder sets 3)

    def __init__(
        self,
        img_size: int,
        output_keys: set[str] | None,
        pad: bool,
        base_transpose: bool = True,
        dtype=torch.bfloat16,
    ):
        self.img_size = img_size
        # A copy, so that the with_* methods do not change the set of the caller.
        self.output_keys: set[str] = set(output_keys or {"img", "cls"})
        self.pad = pad
        self.base_transpose = base_transpose
        self.dtype = dtype
        self._aug: list = []  # augmentation steps accumulated by with_* methods

    def _spatial_keys(self) -> list[str]:
        """Return the image keys that a spatial augmentation changes."""
        return ["img"] + (["mask"] if "mask" in self.output_keys else [])

    def _append_warp(self, warp_cls, **kwargs) -> None:
        """Append a random affine or elastic warp of the image, its mask and its box.

        The box moves with a mask round trip (D-28). ``grid_sample`` fills the
        empty border with 0 only, so the image is shifted by +1 before the warp
        and by -1 after it. Then the background value -1 stays -1.
        """
        keys = self._spatial_keys()
        modes = ["bilinear"] + ["nearest"] * (len(keys) - 1)
        steps = [mn.transforms.Lambdad(keys=["img"], func=_shift_up)]
        if "bbox" in self.output_keys:
            keys, modes = keys + ["bbox_mask"], modes + ["nearest"]
            steps.append(_BboxToMask())
        steps.append(
            warp_cls(keys=keys, mode=modes, padding_mode="zeros", **kwargs)
        )
        if "bbox" in self.output_keys:
            steps.append(_MaskToBbox())
        steps.append(mn.transforms.Lambdad(keys=["img"], func=_shift_down))
        self._aug.append(mn.transforms.Compose(steps))

    # ------------------------------------------------------------------
    # Common augmentations (work for both 2-D and 3-D)
    # ------------------------------------------------------------------

    def with_flip(
        self,
        spatial_axis: int | tuple[int, ...] = 0,
        prob: float = 0.5,
    ) -> "_RadiologyTransformBase":
        """Random flip along one or more spatial axes. The box flips with the image.

        Args:
            spatial_axis: Axis/axes to flip (0 = up-down, 1 = left-right).
            prob: Probability of applying the flip.
        """
        self._aug.append(
            _RandFlipBboxD(keys=self._spatial_keys(), prob=prob, spatial_axis=spatial_axis)
        )
        return self

    def with_affine(
        self,
        translate_range: tuple[float, float] = (10.0, 10.0),
        rotate_range: tuple[float, ...] = (0.175,),
        scale_range: tuple[float, float] = (0.1, 0.1),
        shear_range: tuple[float, ...] = (0.05,),
        prob: float = 0.5,
    ) -> "_RadiologyTransformBase":
        """Random affine transform (translate, rotate, scale, shear) in a single step.

        :meth:`with_rotation` / :meth:`with_zoom` apply each spatial warp
        separately. This step samples all four components together in one
        affine matrix. It is faster, and the interpolation errors do not add up.

        The new box is the box around the warped box (mask round trip). The
        step drops a box that leaves the image, with its label.

        Args:
            translate_range: Maximum translation in pixels per spatial axis
                (x, y) for 2-D, and (x, y, z) for 3-D.
            rotate_range: Maximum rotation in **radians** per rotation plane.
                Pass a single value for 2-D, and three values for 3-D.
                About: 0.175 rad ≈ 10°, 0.35 rad ≈ 20°.
            scale_range: Scale change per axis as a fraction. 0.1 means
                the zoom factor is sampled from ``[0.9, 1.1]``.
            shear_range: Maximum shear per shear plane. Pass a single value
                for 2-D, and up to six values for 3-D.
            prob: Probability of applying the affine transform.
        """
        self._append_warp(
            mn.transforms.RandAffineD,
            translate_range=translate_range,
            rotate_range=rotate_range,
            scale_range=scale_range,
            shear_range=shear_range,
            prob=prob,
        )
        return self

    def with_intensity_jitter(
        self,
        shift: float = 0.1,
        scale: float = 0.1,
        prob: float = 0.5,
    ) -> "_RadiologyTransformBase":
        """Random additive shift and multiplicative scale of pixel intensities.

        Applied independently with the same *prob*:

        * Shift: uniform draw from ``[-shift, +shift]``
        * Scale: uniform draw from ``[1-scale, 1+scale]``

        Args:
            shift: Maximum absolute intensity shift.
            scale: Maximum relative intensity scale.
            prob: Probability of applying each operation.
        """
        self._aug += [
            mn.transforms.RandShiftIntensityD(keys=["img"], offsets=shift, prob=prob),
            mn.transforms.RandScaleIntensityD(keys=["img"], factors=scale, prob=prob),
        ]
        return self

    def with_gaussian_noise(
        self,
        mean: float = 0.0,
        std: float = 0.05,
        prob: float = 0.5,
    ) -> "_RadiologyTransformBase":
        """Add random Gaussian noise to the image.

        Args:
            mean: Mean of the noise distribution.
            std: Standard deviation of the noise.
            prob: Probability of applying the noise.
        """
        self._aug.append(
            mn.transforms.RandGaussianNoiseD(
                keys=["img"], prob=prob, mean=mean, std=std
            )
        )
        return self

    def with_gaussian_smooth(
        self,
        sigma_range: tuple[float, float] = (0.5, 1.5),
        prob: float = 0.2,
    ) -> "_RadiologyTransformBase":
        """Apply random Gaussian smoothing.

        Args:
            sigma_range: Range of Gaussian sigma values.
            prob: Probability of applying the smoothing.
        """
        self._aug.append(
            mn.transforms.RandGaussianSmoothD(
                keys=["img"],
                sigma_x=sigma_range,
                sigma_y=sigma_range,
                prob=prob,
            )
        )
        return self

    def with_elastic_deformation(
        self,
        magnitude_range: tuple[float, float] = (50.0, 150.0),
        spacing: int = 20,
        prob: float = 0.2,
    ) -> "_RadiologyTransformBase":
        """Random elastic deformation (grid-based, 2-D).

        The box moves as in :meth:`with_affine`.

        Args:
            magnitude_range: Magnitude range of the deformation offsets (pixels).
            spacing: Distance in pixels between deformation grid control points.
                Larger values → smoother deformation.
            prob: Probability of applying the deformation.
        """
        self._append_warp(
            mn.transforms.Rand2DElasticD,
            spacing=(spacing, spacing),
            magnitude_range=magnitude_range,
            prob=prob,
        )
        return self

    def with_bbox_as_mask(self, keep_mask: bool = False) -> "_RadiologyTransformBase":
        """Choose if the sample also gives the box as a mask under ``"bbox_mask"``.

        The box moves with every resize, pad, flip, transpose and warp, also
        without this call (D-28). Old code that calls it still works.

        Args:
            keep_mask: If ``True``, the sample also has ``"bbox_mask"``: one
                channel for each box, on the grid of ``"img"``, painted from
                the final boxes. If ``False`` (default), it does not.

        Example::

            transform = (
                RadiologyTransform3D(img_size=112, output_keys={"img", "cls", "bbox"})
                .with_bbox_as_mask(keep_mask=True)
                .with_flip(spatial_axis=0)
                .get_transform()
            )
            # sample → {"img", "cls", "bbox", "bbox_mask"}
        """
        if keep_mask:
            self.output_keys.add("bbox_mask")
        else:
            self.output_keys.discard("bbox_mask")
        return self

    def with_transpose(
        self,
        indices: list[int],
        keys: list[str] | None = None,
    ) -> "_RadiologyTransformBase":
        """Append a spatial transpose step to the augmentation pipeline.

        Applies :class:`monai.transforms.Transposed` with the given *indices*
        after the default preprocessing transpose that corrects axis order on
        load. Use this when your dataset needs a different orientation than
        the default (for example ``[0, 3, 2, 1]`` for 3-D or ``[0, 2, 1]`` for 2-D).

        Args:
            indices: Permutation of axes to pass to ``Transposed``.
                For a 3-D volume with shape ``(C, D, H, W)`` the channel axis
                is ``0``, and the spatial axes are ``1``, ``2``, ``3``.
            keys: Keys to transpose.  Defaults to ``["img"]`` (plus ``"mask"``
                when present in *output_keys*).  Pass an explicit list to
                restrict the transpose to specific keys.  The box moves only
                when ``"img"`` is in *keys*.

        Example::

            # Swap H and W after load for a 2-D image (C, H, W) → (C, W, H)
            transform = (
                RadiologyTransform2D(img_size=224)
                .with_transpose(indices=[0, 2, 1])
                .get_transform()
            )
        """
        # "bbox_mask" does not exist during the augmentation steps (old code passed it).
        keys = [k for k in (keys or self._spatial_keys()) if k != "bbox_mask"]
        self._aug.append(_TransposeBboxD(keys=keys, indices=indices))
        return self

    def add_transform(self, t) -> "_RadiologyTransformBase":
        """Append any custom MONAI transform to the augmentation pipeline.

        This escape hatch lets you insert any transform not covered by the
        built-in ``with_*`` methods.

        Args:
            t: Any MONAI transform (or ``Callable[[dict], dict]``).

        A custom step does not move the box. Use the ``with_*`` methods for
        spatial steps when the sample has a box.

        Example::

            builder.add_transform(
                mn.transforms.RandAffineD(
                    keys=["img"],
                    prob=0.3,
                    translate_range=(10, 10),
                    mode="bilinear",
                )
            )
        """
        self._aug.append(t)
        return self

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def _build_preprocessing(self) -> list:
        raise NotImplementedError

    def _resize_and_pad(self, img_mode: str) -> list:
        """Return the steps that resize the longest edge to ``img_size`` and pad.

        The pad of ``"img"`` also moves the box. The resize needs no box step,
        because a box is a fraction of the image size.
        """
        size = (self.img_size,) * self._NDIM
        t = [
            mn.transforms.ResizeD(
                keys=["img"],
                spatial_size=self.img_size,
                size_mode="longest",
                mode=img_mode,
            )
        ]
        if self.pad:
            t.append(
                _SpatialPadBboxD(
                    keys=["img"],
                    spatial_size=size,
                    mode="constant",
                    constant_values=-1,
                )
            )
        if "mask" in self.output_keys:
            t.append(
                mn.transforms.ResizeD(
                    keys=["mask"],
                    spatial_size=self.img_size,
                    size_mode="longest",
                    mode="nearest",
                )
            )
            if self.pad:
                t.append(
                    mn.transforms.SpatialPadD(
                        keys=["mask"],
                        spatial_size=size,
                        mode="constant",
                        constant_values=0,
                    )
                )
        return t

    def _build_postprocessing(self) -> list:
        # dtype is for the image (and bbox_mask, 0 and 1 only). The labels stay float32, because bfloat16
        # rounds a mask id 257 to 256 and a label 37.3 to 37.25 (V-B-16).
        label_keys = [k for k in ["cls", "reg", "mask"] if k in self.output_keys]
        select_keys = [
            k
            for k in [
                "img",
                "cls",
                "reg",
                "mask",
                "report",
                "bbox",
                "bbox_labels",
                "bbox_findings",
                "bbox_mask",
            ]
            if k in self.output_keys
        ]
        steps = [
            # The box stays float32 for any dtype. bfloat16 keeps 8 bits. So an
            # edge can move by 0.2% of the image size (0.4 px at 224).
            _BboxToTensorD(
                self._NDIM,
                paint_mask="bbox_mask" in self.output_keys,
                mask_dtype=self.dtype,
            ),
        ]
        if label_keys:
            steps.append(mn.transforms.ToTensorD(keys=label_keys, dtype=torch.float32, track_meta=False))
        if "img" in self.output_keys:
            steps.append(mn.transforms.ToTensorD(keys=["img"], dtype=self.dtype, track_meta=False))
        return steps + [mn.transforms.SelectItemsD(keys=select_keys)]

    def get_transform(self) -> mn.transforms.Compose:
        """Assemble and return the final composed transform.

        The pipeline order is:

        1. **Preprocessing**: load, normalize, resize, optional pad (deterministic)
        2. **Augmentations**: all transforms added via ``with_*`` / ``add_transform``
        3. **Postprocessing**: convert to tensor, select output keys

        Returns:
            A :class:`monai.transforms.Compose` object ready to pass to a
            ``Dataset`` or ``PersistentDataset``.
        """
        steps = self._build_preprocessing() + self._aug + self._build_postprocessing()
        return mn.transforms.Compose(steps)


# ---------------------------------------------------------------------------
# 2-D builder
# ---------------------------------------------------------------------------


class RadiologyTransform2D(_RadiologyTransformBase):
    """Fluent transform builder for 2-D radiological images (X-ray, mammogram …).

    Builds a deterministic preprocessing pipeline (load → normalize → resize →
    pad). It is the same pipeline that the built-in dataset classes use.
    Then you can add augmentations on top with the ``with_*`` methods.

    Args:
        img_size: Target spatial size. The longest edge is resized to this
            value. When ``pad=True`` (default) the image is then padded
            with -1 (black, D-37) to ``(img_size, img_size)``. A mask is
            padded with 0. When ``pad=False`` the aspect ratio is preserved
            and the output may not be square.
        output_keys: Set of keys to keep in each sample dict.  Must be a
            subset of ``{"img", "cls", "reg", "mask", "report", "bbox",
            "bbox_labels", "bbox_findings", "bbox_mask"}``.
            Defaults to ``{"img", "cls"}``.
        pad: Whether to pad the resized image to a square.  Set to ``False``
            to keep the original aspect ratio.
        channels: The channel count of ``img`` (D-42). 1 (default) for a gray
            modality: an RGB or RGBA file becomes gray. 3 for color figures.
        dtype: The dtype of ``img`` (default bfloat16). ``cls``, ``reg`` and
            ``mask`` are always float32 (V-B-16).

    Example::

        # Square output (default)
        transform = (
            RadiologyTransform2D(img_size=224, output_keys={"img", "cls"})
            .with_flip(spatial_axis=1)
            .with_affine(translate_range=(10, 10), rotate_range=(0.17,))
            .get_transform()
        )

        # Keep the aspect ratio, with no padding
        transform = (
            RadiologyTransform2D(img_size=224, pad=False)
            .with_intensity_jitter(shift=0.05)
            .get_transform()
        )
    """

    def __init__(
        self,
        img_size: int = 224,
        output_keys: set[str] | None = None,
        pad: bool = True,
        base_transpose: bool = True,
        dtype=torch.bfloat16,
        channels: int = 1,
    ):
        super().__init__(img_size, output_keys, pad, base_transpose, dtype)
        self.channels = channels

    def _build_preprocessing(self) -> list:
        t = [_ParseBboxD(self._NDIM)]
        t += _base_load_2d(self._spatial_keys(), base_transpose=self.base_transpose, channels=self.channels)
        if "mask" in self.output_keys:
            t.append(mn.transforms.Lambdad(keys=["mask"], func=_binary_mask))
        t += self._resize_and_pad("bilinear")
        return t


# ---------------------------------------------------------------------------
# 3-D builder
# ---------------------------------------------------------------------------


class RadiologyTransform3D(_RadiologyTransformBase):
    """Fluent transform builder for 3-D radiological volumes (CT, MRI …).

    Args:
        img_size: Isotropic target size. Volumes are resized so the longest
            spatial dimension equals ``img_size``. When ``pad=True`` (default)
            the volume is then padded with -1 (black, D-37) to
            ``(img_size, img_size, img_size)``. A mask is padded with 0. When
            ``pad=False`` the output may not be a cube.
        output_keys: Set of keys to keep.  Defaults to ``{"img", "cls"}``.
        hu_window: Optional ``(min_HU, max_HU)`` clipping window applied
            before percentile normalization (useful for CT).
        pad: Whether to pad the resized volume to a cube.  Set to ``False``
            to preserve the original aspect ratio.
        pixdim: Optional ``(x, y, z)`` target voxel spacing in mm.  When set,
            SpacingD resamples each volume to this spacing before Resize/Pad.
            Use :meth:`with_spacing` for the fluent-API equivalent.
        dtype: The dtype of ``img`` (default bfloat16). ``cls``, ``reg`` and
            ``mask`` are always float32 (V-B-16).

    Example::

        # Isotropic 1 mm cube (explicit resampling)
        transform = (
            RadiologyTransform3D(img_size=112, hu_window=(-1000, 400))
            .with_spacing((1.0, 1.0, 1.0))
            .with_flip(spatial_axis=0)
            .with_affine(translate_range=(5, 5, 5), rotate_range=(0.09, 0.09, 0.09))
            .get_transform()
        )

        # No padding, so it keeps the aspect ratio
        transform = (
            RadiologyTransform3D(img_size=112, pad=False)
            .get_transform()
        )
    """

    _NDIM = 3

    def __init__(
        self,
        img_size: int = 112,
        output_keys: set[str] | None = None,
        hu_window: tuple[float, float] | None = None,
        pad: bool = True,
        base_transpose: bool = True,
        pixdim: tuple[float, float, float] | None = None,
        axcodes: str = "IPL",
        dtype=torch.bfloat16,
    ):
        super().__init__(img_size, output_keys, pad, base_transpose, dtype)
        self.hu_window = hu_window
        self.pixdim = pixdim
        self.axcodes = axcodes

    def with_spacing(
        self,
        pixdim: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> "RadiologyTransform3D":
        """Resample to a target voxel spacing (mm) before Resize/Pad.

        This is critical for anisotropic MRI acquisitions (for example lumbar
        spine at 0.5 x 0.5 x 4 mm). Without it, ``ResizeD(size_mode="longest")``
        collapses the through-plane axis into a few slices. Then the
        coronal/sagittal views look squeezed. ITKReader makes the
        MetaTensor affine from the DICOM headers (PixelSpacing, and the slice
        positions along z), so DICOM series need no extra step. The box moves
        with the resample (D-28).

        Args:
            pixdim: Target spacing in mm per spatial axis. ``(1, 1, 1)``
                gives 1 mm isotropic. ``(2, 2, 2)`` is a coarser target,
                useful when memory is low. An anisotropic target (for
                example ``(1.5, 1.5, 0.5)``) keeps through-plane detail,
                but loses in-plane resolution.
        """
        self.pixdim = tuple(float(x) for x in pixdim)
        return self

    def with_gaussian_smooth(
        self,
        sigma_range: tuple[float, float] = (0.5, 1.5),
        prob: float = 0.2,
    ) -> "RadiologyTransform3D":
        """Apply random Gaussian smoothing (3-D variant).

        Args:
            sigma_range: Range of sigma values for all three axes.
            prob: Probability of applying the smoothing.
        """
        self._aug.append(
            mn.transforms.RandGaussianSmoothD(
                keys=["img"],
                sigma_x=sigma_range,
                sigma_y=sigma_range,
                sigma_z=sigma_range,
                prob=prob,
            )
        )
        return self

    def with_elastic_deformation(
        self,
        sigma_range: tuple[float, float] = (5.0, 7.0),
        magnitude_range: tuple[float, float] = (50.0, 150.0),
        prob: float = 0.2,
    ) -> "RadiologyTransform3D":
        """Random elastic deformation (3-D grid-based).

        The box moves as in :meth:`with_affine`.

        Args:
            sigma_range: Smoothing sigma for the deformation field.
            magnitude_range: Magnitude range of the deformation.
            prob: Probability of applying the deformation.
        """
        self._append_warp(
            mn.transforms.Rand3DElasticD,
            sigma_range=sigma_range,
            magnitude_range=magnitude_range,
            prob=prob,
        )
        return self

    def _build_preprocessing(self) -> list:
        t = [_ParseBboxD(self._NDIM)]
        t += _base_load_3d(
            self._spatial_keys(),
            self.hu_window,
            base_transpose=self.base_transpose,
            pixdim=self.pixdim,
            axcodes=self.axcodes,
        )
        t += self._resize_and_pad("trilinear")
        return t
