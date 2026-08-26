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

import torch
import monai as mn

from radharmony.dataset.base import _apply_voi_lut, _convert_monochrome1_to_2


# ---------------------------------------------------------------------------
# Bbox → mask helper
# ---------------------------------------------------------------------------


class _MaskToBbox:
    """Convert an augmented ``"bbox_mask"`` back to normalized bbox coordinates.

    When the mask has **multiple channels** (one per bbox, written by
    ``_BboxToMask``), each channel is converted independently and the
    result is a list of ``[dim0_min, dim0_max, dim1_min, dim1_max]``
    lists.  A single-channel mask produces a single flat list (backwards
    compatible).  Channels that are all-zero (bbox augmented out of
    frame) are dropped.
    """

    @staticmethod
    def _channel_to_bbox(chan, spatial):
        """Extract normalized bbox coords from one (H, W) or (D, H, W) mask."""
        nonzero = torch.nonzero(chan, as_tuple=False)
        if nonzero.numel() == 0:
            return None
        coords: list[float] = []
        for i, dim_size in enumerate(spatial):
            coords.append(nonzero[:, i].min().item() / dim_size)
            coords.append(nonzero[:, i].max().item() / dim_size)
        return coords

    def __call__(self, data: dict) -> dict:
        if "bbox_mask" not in data:
            return data
        mask = data["bbox_mask"]  # (C, *spatial)
        n_channels = mask.shape[0]
        spatial = mask.shape[1:]

        # Parse JSON-string per-box lists (some harmonizers store as JSON).
        # bbox_findings is an optional parallel list (e.g. Chest ImaGenome).
        import json

        def _parse_list(key):
            raw = data.get(key)
            if isinstance(raw, str):
                try:
                    data[key] = json.loads(raw)
                except (ValueError, TypeError):
                    data[key] = []

        _parse_list("bbox_labels")
        _parse_list("bbox_findings")
        has_labels = "bbox_labels" in data and data["bbox_labels"]
        has_findings = "bbox_findings" in data and data["bbox_findings"]

        if n_channels == 1:
            # Single bbox — return flat list (backwards compatible)
            coords = self._channel_to_bbox(mask[0], spatial)
            data["bbox"] = coords if coords else [0.0] * (2 * len(spatial))
        else:
            # Multi-bbox — return list of lists, drop empty channels.  Keep any
            # parallel per-box lists (labels, findings) index-aligned with the
            # boxes that survive augmentation.
            boxes = []
            kept_labels = []
            kept_findings = []
            for c in range(n_channels):
                coords = self._channel_to_bbox(mask[c], spatial)
                if coords is not None:
                    boxes.append(coords)
                    if has_labels and c < len(data["bbox_labels"]):
                        kept_labels.append(data["bbox_labels"][c])
                    if has_findings and c < len(data["bbox_findings"]):
                        kept_findings.append(data["bbox_findings"][c])
            data["bbox"] = boxes if boxes else [[0.0] * (2 * len(spatial))]
            if has_labels:
                data["bbox_labels"] = kept_labels
            if has_findings:
                data["bbox_findings"] = kept_findings

        return data


class _BboxToMask:
    """Convert normalized bbox(es) to a binary spatial mask.

    Accepts either a single flat bbox::

        [dim0_min, dim0_max, dim1_min, dim1_max, ...]   (all in [0, 1])

    or a list of such bboxes (multi-bbox per image)::

        [[y_min, y_max, x_min, x_max], [y_min, y_max, x_min, x_max], ...]

    For a single bbox, produces a ``(1, *spatial)`` mask.  For multiple
    bboxes, produces a ``(N, *spatial)`` mask with one channel per bbox
    so that ``_MaskToBbox`` can recover individual boxes after augmentation.
    """

    @staticmethod
    def _paint_one_bbox(mask, coords, spatial):
        """Fill a single normalized bbox into *mask* (mutated in-place)."""
        ndim = len(spatial)
        if len(coords) < 2 * ndim:
            return
        slices: list = [slice(None)]  # channel axis
        for i, dim_size in enumerate(spatial):
            lo = int(coords[2 * i] * dim_size)
            hi = int(coords[2 * i + 1] * dim_size + 0.5)
            hi = max(lo + 1, min(hi, dim_size))  # at least 1 voxel, clamp
            slices.append(slice(lo, hi))
        mask[tuple(slices)] = 1

    def __call__(self, data: dict) -> dict:
        if "bbox" not in data:
            return data
        img = data["img"]
        bbox = data["bbox"]
        spatial = img.shape[1:]  # drop channel dim

        # JSON strings (from harmonizers that store bbox as a JSON column) must
        # be parsed to a list before any coordinate logic runs.
        if isinstance(bbox, str):
            import json
            try:
                bbox = json.loads(bbox)
            except (ValueError, TypeError):
                return data

        # Detect multi-bbox (list of lists) vs single bbox (flat list/tensor)
        is_multi = (
            isinstance(bbox, (list, tuple))
            and bbox
            and isinstance(bbox[0], (list, tuple))
        )

        if is_multi:
            # One channel per bbox so individual boxes survive augmentation
            mask = torch.zeros(len(bbox), *spatial, dtype=img.dtype)
            for c, single in enumerate(bbox):
                try:
                    coords = [float(v) for v in single]
                    if coords and all(0.0 <= v <= 1.0 for v in coords):
                        one_ch = mask[c : c + 1]
                        self._paint_one_bbox(one_ch, coords, spatial)
                except (TypeError, ValueError):
                    continue
        else:
            mask = torch.zeros(1, *spatial, dtype=img.dtype)
            try:
                coords = [float(v) for v in bbox]
                if coords and all(0.0 <= v <= 1.0 for v in coords):
                    self._paint_one_bbox(mask, coords, spatial)
            except (TypeError, ValueError):
                pass

        data["bbox_mask"] = mask
        return data


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


class _ReorientBbox:
    """Permute bbox coord pairs to stay aligned after OrientationD(axcodes)+TransposeD([0,3,2,1]).

    Must run AFTER EnsureChannelFirstD (affine populated) and BEFORE OrientationD.
    Uses SimpleITK to read the source orientation code from the image direction
    cosines — no nibabel dependency.  No-op when ``"bbox"`` is absent.

    Args:
        axcodes: Target orientation string (e.g. ``"RAS"``, ``"LPS"``, ``"IPL"``).
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

        (``src_code`` here comes from the helper below, which builds direction
        cosines for the SITK image used by ``_SITKOrientD``; that string is
        the *reversed* of the MONAI-standard axcodes for the affine, which is
        why the formula uses ``sigma`` directly rather than ``2 - sigma[2-i]``.)
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

        try:
            import numpy as np

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

            bbox = data["bbox"]
            is_multi = (
                isinstance(bbox, (list, tuple))
                and len(bbox) > 0
                and isinstance(bbox[0], (list, tuple))
            )
            data["bbox"] = (
                [_xfm(list(b)) for b in bbox] if is_multi else _xfm(list(bbox))
            )
        except Exception:
            pass

        return data


class _SITKOrientD:
    """Reorient 3-D MetaTensors to a canonical orientation using SimpleITK.

    Drop-in replacement for ``mn.transforms.OrientationD`` that uses
    SimpleITK's ``DICOMOrientImageFilter`` instead of nibabel.  Each key
    must be a MetaTensor with a populated RAS affine (produced by ITKReader +
    EnsureChannelFirstD).  Keys without a MetaTensor affine are skipped.

    Axis-reversal convention (MONAI spatial dim j = ITK axis 2-j):
    - ``tensor[0].numpy()`` shape ``(D, H, W)`` is passed directly to
      ``sitk.GetImageFromArray``, which treats it as ``(z, y, x)`` — correct.
    - After reorientation, ``sitk.GetArrayFromImage`` returns ``(sz, sy, sx)``
      which maps back to MONAI ``(dim0, dim1, dim2)`` — also correct.

    Args:
        keys: Data dict keys to reorient.
        axcodes: Target 3-letter orientation string (e.g. ``"RAS"``, ``"LPS"``).
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


def _base_load_2d(img_keys: list[str], base_transpose: bool = True) -> list:
    """Return the load + normalise steps shared by all 2-D pipelines."""
    t = [
        mn.transforms.LoadImageD(keys=img_keys, reader="ITKReader"),
        mn.transforms.EnsureChannelFirstD(keys=img_keys),
        # ITKReader loads 2-D DICOMs as (H, W, 1); after EnsureChannelFirst → (C, H, W, 1).
        # Non-DICOM 2-D images (JPEG, PNG) load as (C, H, W).
        mn.transforms.Lambdad(
            keys=img_keys, func=lambda x: x[..., 0] if x.ndim == 4 else x
        ),
    ]
    if base_transpose:
        # ITK axis correction: not needed when loading numpy arrays directly.
        t.append(mn.transforms.Transposed(keys=img_keys, indices=[0, 2, 1]))
    t += [
        mn.transforms.Lambdad(keys=["img"], func=_apply_voi_lut),
        # mn.transforms.Lambdad(keys=["img"], func=_convert_monochrome1_to_2),
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
    """Return the load + normalise steps shared by all 3-D pipelines.

    When ``pixdim`` is set, SpacingD resamples to that target voxel spacing
    (mm) using the MetaTensor affine populated by ITKReader from DICOM
    PixelSpacing + SliceThickness.  Critical for anisotropic MRI — without
    it, e.g. 0.5 x 0.5 x 4 mm lumbar-spine series get flattened by the
    longest-axis ResizeD into a ~7-slice pancake.

    ``axcodes`` controls the canonical orientation target for ``OrientationD``
    and the matching ``_ReorientBbox`` bbox-coordinate fix.  Must match
    across both transforms (default ``"RAS"``).
    """
    t = [
        mn.transforms.LoadImageD(keys=img_keys, reader="ITKReader"),
        mn.transforms.EnsureChannelFirstD(keys=img_keys),
    ]
    if base_transpose:
        # Reorient bbox coords BEFORE OrientationD so the coordinate pairs
        # stay aligned with the reoriented image (see _ReorientBbox docstring).
        t.append(_ReorientBbox(axcodes=axcodes))
        # Normalise scanner axis order to a canonical orientation so that all
        # studies load with the same axis semantics regardless of patient
        # position or acquisition convention.
        t.append(_SITKOrientD(keys=img_keys, axcodes=axcodes))
    if pixdim is not None:
        # Bilinear for intensity, nearest for masks (keeps labels discrete).
        spacing_modes = tuple("bilinear" if k == "img" else "nearest" for k in img_keys)
        t.append(
            mn.transforms.SpacingD(
                keys=img_keys,
                pixdim=pixdim,
                mode=spacing_modes,
            )
        )
    if base_transpose:
        # ITK axis correction: not needed when loading numpy arrays directly.
        #
        # ⚠ DO NOT REMOVE without reading docs/proposals/orientation_normalization.md
        # AND running tests/test_bbox_axis_alignment.py.  This is NOT just
        # cosmetic — bbox harmonizers (see commit d1d0535 for the lumbar spine
        # one) author bbox coordinates in the POST-transpose (D, H, W) axis
        # order.  _ReorientBbox (above) pre-adjusts the bbox coords so they
        # remain correct after this transpose.
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

    Do not instantiate directly — use :class:`RadiologyTransform2D` or
    :class:`RadiologyTransform3D`.
    """

    def __init__(
        self,
        img_size: int,
        output_keys: set[str] | None,
        pad: bool,
        base_transpose: bool = True,
        dtype=torch.bfloat16,
    ):
        self.img_size = img_size
        self.output_keys: set[str] = output_keys or {"img", "cls"}
        self.pad = pad
        self.base_transpose = base_transpose
        self.dtype = dtype
        self._aug: list = []  # augmentation steps accumulated by with_* methods

    # ------------------------------------------------------------------
    # Common augmentations (work for both 2-D and 3-D)
    # ------------------------------------------------------------------

    def with_flip(
        self,
        spatial_axis: int | tuple[int, ...] = 0,
        prob: float = 0.5,
    ) -> "_RadiologyTransformBase":
        """Random flip along one or more spatial axes.

        Args:
            spatial_axis: Axis/axes to flip (0 = up-down, 1 = left-right).
            prob: Probability of applying the flip.
        """
        keys = (
            ["img"]
            + (["mask"] if "mask" in self.output_keys else [])
            + (["bbox_mask"] if "bbox_mask" in self.output_keys else [])
        )
        self._aug.append(
            mn.transforms.RandFlipD(keys=keys, prob=prob, spatial_axis=spatial_axis)
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

        Unlike :meth:`with_rotation` / :meth:`with_zoom` which apply each
        spatial warp separately, this samples all four components jointly from
        one affine matrix, which is more efficient and avoids compounding
        interpolation errors.

        Args:
            translate_range: Maximum translation in pixels per spatial axis
                (x, y) for 2-D; (x, y, z) for 3-D.
            rotate_range: Maximum rotation in **radians** per rotation plane.
                Pass a single value for 2-D; three values for 3-D.
                Approximately: 0.175 rad ≈ 10°, 0.35 rad ≈ 20°.
            scale_range: Scale perturbation per axis as a fraction; 0.1 means
                the zoom factor is sampled from ``[0.9, 1.1]``.
            shear_range: Maximum shear per shear plane. Pass a single value
                for 2-D; up to six values for 3-D.
            prob: Probability of applying the affine transform.
        """
        has_mask = "mask" in self.output_keys
        has_bbox_mask = "bbox_mask" in self.output_keys
        keys = (
            ["img"]
            + (["mask"] if has_mask else [])
            + (["bbox_mask"] if has_bbox_mask else [])
        )
        modes = (
            ["bilinear"]
            + (["nearest"] if has_mask else [])
            + (["nearest"] if has_bbox_mask else [])
        )
        self._aug.append(
            mn.transforms.Compose(
                [
                    mn.transforms.Lambdad(keys=["img"], func=lambda x: x + 1),
                    mn.transforms.RandAffineD(
                        keys=keys,
                        translate_range=translate_range,
                        rotate_range=rotate_range,
                        scale_range=scale_range,
                        shear_range=shear_range,
                        prob=prob,
                        mode=modes,
                        padding_mode="zeros",
                    ),
                    mn.transforms.Lambdad(keys=["img"], func=lambda x: x - 1),
                ]
            )
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

        Args:
            magnitude_range: Magnitude range of the deformation offsets (pixels).
            spacing: Distance in pixels between deformation grid control points.
                Larger values → smoother deformation.
            prob: Probability of applying the deformation.
        """
        # grid_sample only supports "zeros"/"border"/"reflection", not arbitrary
        # constants. Shift by +1 so the true background value (-1) maps to 0,
        # apply the deformation, then shift back.
        has_mask = "mask" in self.output_keys
        has_bbox_mask = "bbox_mask" in self.output_keys
        keys = (
            ["img"]
            + (["mask"] if has_mask else [])
            + (["bbox_mask"] if has_bbox_mask else [])
        )
        modes = (
            ["bilinear"]
            + (["nearest"] if has_mask else [])
            + (["nearest"] if has_bbox_mask else [])
        )
        self._aug.append(
            mn.transforms.Compose(
                [
                    mn.transforms.Lambdad(keys=["img"], func=lambda x: x + 1),
                    mn.transforms.Rand2DElasticD(
                        keys=keys,
                        spacing=(spacing, spacing),
                        magnitude_range=magnitude_range,
                        prob=prob,
                        mode=modes,
                        padding_mode="zeros",
                    ),
                    mn.transforms.Lambdad(keys=["img"], func=lambda x: x - 1),
                ]
            )
        )
        return self

    def with_bbox_as_mask(self, keep_mask: bool = False) -> "_RadiologyTransformBase":
        """Convert the bbox to a binary spatial mask stored under ``"bbox_mask"``.

        The mask is created after the image is resized (so coordinates map to
        the correct pixel grid) but **before** padding (so the mask is padded
        with zeros consistently with the image).  All subsequent spatial
        augmentations that include ``"bbox_mask"`` in their keys will keep the
        mask aligned with the image.  After augmentation, ``"bbox_mask"`` is
        converted back to normalized bbox coordinates in ``"bbox"``.

        Can coexist with a real segmentation mask under ``"mask"``.  Requires
        ``"bbox"`` in *output_keys*.  Automatically adds ``"bbox_mask"`` to
        *output_keys*.

        Args:
            keep_mask: If ``False`` (default), ``"bbox_mask"`` is dropped once
                the updated ``"bbox"`` coordinates have been extracted.  Set to
                ``True`` to keep it in the output alongside ``"bbox"``.

        Example::

            # Drop bbox_mask after back-conversion (default)
            transform = (
                RadiologyTransform3D(img_size=112, output_keys={"img", "cls", "bbox"})
                .with_bbox_as_mask()
                .with_flip(spatial_axis=0)
                .get_transform()
            )
            # sample → {"img", "cls", "bbox"}

            # Keep bbox_mask in output
            transform = (
                RadiologyTransform3D(img_size=112, output_keys={"img", "cls", "bbox"})
                .with_bbox_as_mask(keep_mask=True)
                .get_transform()
            )
            # sample → {"img", "cls", "bbox", "bbox_mask"}
        """
        self._bbox_as_mask = True
        self._bbox_mask_keep = keep_mask
        self.output_keys.add("bbox_mask")
        return self

    def with_transpose(
        self,
        indices: list[int],
        keys: list[str] | None = None,
    ) -> "_RadiologyTransformBase":
        """Append a spatial transpose step to the augmentation pipeline.

        Applies :class:`monai.transforms.Transposed` with the given *indices*
        after the default preprocessing transpose that corrects axis order on
        load.  Use this when your dataset requires a different orientation than
        the default (e.g. ``[0, 3, 2, 1]`` for 3-D or ``[0, 2, 1]`` for 2-D).

        Args:
            indices: Permutation of axes to pass to ``Transposed``.
                For a 3-D volume with shape ``(C, D, H, W)`` the channel axis
                is ``0``; spatial axes are ``1``, ``2``, ``3``.
            keys: Keys to transpose.  Defaults to ``["img"]`` (plus ``"mask"``
                and ``"bbox_mask"`` when present in *output_keys*).  Pass an
                explicit list to restrict the transpose to specific keys —
                e.g. ``keys=["img"]`` when the bbox mask is already in the
                correct orientation and should not be transposed.

        Example::

            # Swap H and W after load for a 2-D image (C, H, W) → (C, W, H)
            transform = (
                RadiologyTransform2D(img_size=224)
                .with_transpose(indices=[0, 2, 1])
                .get_transform()
            )
        """
        if keys is None:
            keys = (
                ["img"]
                + (["mask"] if "mask" in self.output_keys else [])
                + (["bbox_mask"] if "bbox_mask" in self.output_keys else [])
            )
        self._aug.append(mn.transforms.Transposed(keys=keys, indices=indices))
        return self

    def add_transform(self, t) -> "_RadiologyTransformBase":
        """Append any custom MONAI transform to the augmentation pipeline.

        This escape hatch lets you insert any transform not covered by the
        built-in ``with_*`` methods.

        Args:
            t: Any MONAI transform (or ``Callable[[dict], dict]``).

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

    def _build_postprocessing(self) -> list:
        raise NotImplementedError

    def get_transform(self) -> mn.transforms.Compose:
        """Assemble and return the final composed transform.

        The pipeline order is:

        1. **Preprocessing** — load, normalise, resize, optional pad (deterministic)
        2. **Augmentations** — all transforms added via ``with_*`` / ``add_transform``
        3. **Postprocessing** — convert to tensor, select output keys

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

    Builds a deterministic preprocessing pipeline (load → normalise → resize →
    pad) identical to the one used internally by the built-in dataset classes,
    then lets you layer augmentations on top via the ``with_*`` methods.

    Args:
        img_size: Target spatial size; the longest edge is resized to this
            value.  When ``pad=True`` (default) the image is then zero-padded
            to ``(img_size, img_size)``; when ``pad=False`` the aspect ratio
            is preserved and the output may not be square.
        output_keys: Set of keys to keep in each sample dict.  Must be a
            subset of ``{"img", "cls", "mask", "report", "bbox"}``.
            Defaults to ``{"img", "cls"}``.
        pad: Whether to pad the resized image to a square.  Set to ``False``
            to keep the original aspect ratio.

    Example::

        # Square output (default)
        transform = (
            RadiologyTransform2D(img_size=224, output_keys={"img", "cls"})
            .with_flip(spatial_axis=1)
            .with_affine(translate_range=(10, 10), rotate_range=(0.17,))
            .get_transform()
        )

        # Preserve aspect ratio — no padding
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
    ):
        super().__init__(img_size, output_keys, pad, base_transpose, dtype)

    def _build_preprocessing(self) -> list:
        bbox_as_mask = getattr(self, "_bbox_as_mask", False)
        img_keys = ["img"] + (["mask"] if "mask" in self.output_keys else [])
        t = _base_load_2d(img_keys, base_transpose=self.base_transpose)
        # Convert bbox → bbox_mask at original image size before any resizing,
        # so ResizeD (nearest) handles the spatial scaling consistently with
        # how segmentation masks are treated.
        if bbox_as_mask:
            t.append(_BboxToMask())
        t.append(
            mn.transforms.ResizeD(
                keys=["img"],
                spatial_size=self.img_size,
                size_mode="longest",
                mode="bilinear",
            )
        )
        if self.pad:
            t.append(
                mn.transforms.SpatialPadD(
                    keys=["img"],
                    spatial_size=(self.img_size, self.img_size),
                    mode="constant",
                    constant_values=-1,
                )
            )
        for mask_key in (["mask"] if "mask" in self.output_keys else []) + (
            ["bbox_mask"] if bbox_as_mask else []
        ):
            t.append(
                mn.transforms.ResizeD(
                    keys=[mask_key],
                    spatial_size=self.img_size,
                    size_mode="longest",
                    mode="nearest",
                )
            )
            if self.pad:
                t.append(
                    mn.transforms.SpatialPadD(
                        keys=[mask_key],
                        spatial_size=(self.img_size, self.img_size),
                        mode="constant",
                        constant_values=0,
                    )
                )
        return t

    def _build_postprocessing(self) -> list:
        t = []
        if getattr(self, "_bbox_as_mask", False):
            t.append(_MaskToBbox())
        keep_bbox_mask = getattr(self, "_bbox_mask_keep", True)
        out_keys = (
            self.output_keys if keep_bbox_mask else self.output_keys - {"bbox_mask"}
        )
        tensor_keys = [
            k for k in ["img", "cls", "reg", "mask", "bbox", "bbox_mask"] if k in out_keys
        ]
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
            if k in out_keys
        ]
        t += [
            mn.transforms.ToTensorD(
                keys=tensor_keys, dtype=self.dtype, track_meta=False
            ),
            mn.transforms.SelectItemsD(keys=select_keys),
        ]
        return t


# ---------------------------------------------------------------------------
# 3-D builder
# ---------------------------------------------------------------------------


class RadiologyTransform3D(_RadiologyTransformBase):
    """Fluent transform builder for 3-D radiological volumes (CT, MRI …).

    Args:
        img_size: Isotropic target size; volumes are resized so the longest
            spatial dimension equals ``img_size``.  When ``pad=True`` (default)
            the volume is then zero-padded to
            ``(img_size, img_size, img_size)``; when ``pad=False`` the output
            may not be a cube.
        output_keys: Set of keys to keep.  Defaults to ``{"img", "cls"}``.
        hu_window: Optional ``(min_HU, max_HU)`` clipping window applied
            before percentile normalisation (useful for CT).
        pad: Whether to pad the resized volume to a cube.  Set to ``False``
            to preserve the original aspect ratio.
        pixdim: Optional ``(x, y, z)`` target voxel spacing in mm.  When set,
            SpacingD resamples each volume to this spacing before Resize/Pad.
            Use :meth:`with_spacing` for the fluent-API equivalent.

    Example::

        # Isotropic 1 mm cube (explicit resampling)
        transform = (
            RadiologyTransform3D(img_size=112, hu_window=(-1000, 400))
            .with_spacing((1.0, 1.0, 1.0))
            .with_flip(spatial_axis=0)
            .with_affine(translate_range=(5, 5, 5), rotate_range=(0.09, 0.09, 0.09))
            .get_transform()
        )

        # No padding — keeps aspect ratio
        transform = (
            RadiologyTransform3D(img_size=112, pad=False)
            .get_transform()
        )
    """

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

        Critical for anisotropic MRI acquisitions (e.g. lumbar spine at
        0.5 x 0.5 x 4 mm): without this, ``ResizeD(size_mode="longest")``
        collapses the through-plane axis into a few slices, leaving the
        coronal/sagittal views visually squeezed.  ITKReader populates the
        MetaTensor affine from DICOM PixelSpacing + SliceThickness, so no
        extra plumbing is needed on DICOM series.

        Args:
            pixdim: Target spacing in mm per spatial axis.  ``(1, 1, 1)``
                gives 1 mm isotropic; ``(2, 2, 2)`` is a coarser target
                useful when memory is tight.  Anisotropic targets
                (e.g. ``(1.5, 1.5, 0.5)``) can be used to preserve
                through-plane detail at the cost of in-plane resolution.
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

        Args:
            sigma_range: Smoothing sigma for the deformation field.
            magnitude_range: Magnitude range of the deformation.
            prob: Probability of applying the deformation.
        """
        # Same +1/-1 shift workaround as the 2-D variant (see above).
        has_mask = "mask" in self.output_keys
        has_bbox_mask = "bbox_mask" in self.output_keys
        keys = (
            ["img"]
            + (["mask"] if has_mask else [])
            + (["bbox_mask"] if has_bbox_mask else [])
        )
        modes = (
            ["bilinear"]
            + (["nearest"] if has_mask else [])
            + (["nearest"] if has_bbox_mask else [])
        )
        self._aug.append(
            mn.transforms.Compose(
                [
                    mn.transforms.Lambdad(keys=["img"], func=lambda x: x + 1),
                    mn.transforms.Rand3DElasticD(
                        keys=keys,
                        sigma_range=sigma_range,
                        magnitude_range=magnitude_range,
                        prob=prob,
                        mode=modes,
                        padding_mode="zeros",
                    ),
                    mn.transforms.Lambdad(keys=["img"], func=lambda x: x - 1),
                ]
            )
        )
        return self

    def _build_preprocessing(self) -> list:
        bbox_as_mask = getattr(self, "_bbox_as_mask", False)
        img_keys = ["img"] + (["mask"] if "mask" in self.output_keys else [])
        t = _base_load_3d(
            img_keys,
            self.hu_window,
            base_transpose=self.base_transpose,
            pixdim=self.pixdim,
            axcodes=self.axcodes,
        )
        size3 = (self.img_size, self.img_size, self.img_size)
        # Convert bbox → bbox_mask at original image size before any resizing,
        # so ResizeD (nearest) handles the spatial scaling consistently with
        # how segmentation masks are treated.
        if bbox_as_mask:
            t.append(_BboxToMask())
        t.append(
            mn.transforms.ResizeD(
                keys=["img"],
                spatial_size=self.img_size,
                size_mode="longest",
                mode="trilinear",
            )
        )
        if self.pad:
            t.append(
                mn.transforms.SpatialPadD(
                    keys=["img"],
                    spatial_size=size3,
                    mode="constant",
                    constant_values=-1,
                )
            )
        for mask_key in (["mask"] if "mask" in self.output_keys else []) + (
            ["bbox_mask"] if bbox_as_mask else []
        ):
            t.append(
                mn.transforms.ResizeD(
                    keys=[mask_key],
                    spatial_size=self.img_size,
                    size_mode="longest",
                    mode="nearest",
                )
            )
            if self.pad:
                t.append(
                    mn.transforms.SpatialPadD(
                        keys=[mask_key],
                        spatial_size=size3,
                        mode="constant",
                        constant_values=0,
                    )
                )
        return t

    def _build_postprocessing(self) -> list:
        t = []
        if getattr(self, "_bbox_as_mask", False):
            t.append(_MaskToBbox())
        keep_bbox_mask = getattr(self, "_bbox_mask_keep", True)
        out_keys = (
            self.output_keys if keep_bbox_mask else self.output_keys - {"bbox_mask"}
        )
        tensor_keys = [
            k for k in ["img", "cls", "reg", "mask", "bbox", "bbox_mask"] if k in out_keys
        ]
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
            if k in out_keys
        ]
        t += [
            mn.transforms.ToTensorD(
                keys=tensor_keys, dtype=self.dtype, track_meta=False
            ),
            mn.transforms.SelectItemsD(keys=select_keys),
        ]
        return t
