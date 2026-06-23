# Transforms

RadHarmony provides two transform classes: `RadiologyTransform2D` and `RadiologyTransform3D`. Both use a fluent builder pattern and wrap a MONAI `Compose` pipeline.

## Default 2D pipeline

```
LoadImageD → EnsureChannelFirstD → VOI LUT (DICOM) →
ScaleIntensityPercentilesD → [augmentations] → ToTensorD
```

```python
from radharmony.dataset.transforms import RadiologyTransform2D
import torch

t = RadiologyTransform2D(img_size=224, dtype=torch.float32).get_transform()
```

## Default 3D pipeline

```
LoadImageD (ITKReader) → EnsureChannelFirstD → _ReorientBbox (if bbox) →
_SITKOrientD (axcodes, default "IPL") → SpacingD (if pixdim) →
TransposeD ([0,3,2,1]) → ScaleIntensityRangeD (HU clip, if hu_window) →
ScaleIntensityPercentilesD → [augmentations] → Resize/Pad → ToTensorD
```

```python
from radharmony.dataset.transforms import RadiologyTransform3D

t = RadiologyTransform3D(
    img_size=112,
    hu_window=(-1000, 400),   # CT lung window
    pixdim=(1.5, 1.5, 1.5),
    dtype=torch.float32,
).get_transform()
```

---

## Constructor arguments

### `RadiologyTransform2D`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `img_size` | `int` | `224` | Square output size in pixels |
| `output_keys` | `set[str]` | `None` | Keys to transform; auto-detected from dataset if `None` |
| `pad` | `bool` | `True` | Pad to square before resizing |
| `base_transpose` | `bool` | `True` | Apply default axis transpose |
| `dtype` | `torch.dtype` | `torch.bfloat16` | Output tensor dtype |

### `RadiologyTransform3D`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `img_size` | `int` | `112` | Isotropic output size in voxels |
| `output_keys` | `set[str]` | `None` | Keys to transform |
| `hu_window` | `tuple[float, float] \| None` | `None` | HU clip range, e.g. `(-1000, 400)`; `None` = no clip (use for MRI) |
| `pad` | `bool` | `True` | Pad to cube before resizing |
| `base_transpose` | `bool` | `True` | Apply orientation normalisation + `[0,3,2,1]` transpose (load-bearing for bbox alignment) |
| `pixdim` | `tuple[float,float,float] \| None` | `None` | Target voxel spacing in mm; `None` = no resampling |
| `axcodes` | `str` | `"IPL"` | Canonical orientation target (e.g. `"RAS"`, `"LPS"`, `"IPL"`); applies to both `_SITKOrientD` and `_ReorientBbox` |
| `dtype` | `torch.dtype` | `torch.bfloat16` | Output tensor dtype |

---

## Fluent builder: adding augmentations

Chain augmentation methods before calling `.get_transform()`:

```python
from radharmony.dataset.transforms import RadiologyTransform2D
import torch

t = (
    RadiologyTransform2D(img_size=224, dtype=torch.float32)
    .with_flip(spatial_axis=0, prob=0.5)
    .with_affine(translate_range=(10.0, 10.0), rotate_range=(0.175,), prob=0.5)
    .with_intensity_jitter(shift=0.1, scale=0.1, prob=0.5)
    .with_gaussian_noise(mean=0.0, std=0.05, prob=0.5)
    .get_transform()
)
```

### Available augmentation methods

| Method | Parameters | Notes |
|--------|-----------|-------|
| `with_flip(spatial_axis, prob)` | axis: `int\|tuple`; prob: `float` | Random mirror flip |
| `with_affine(translate_range, rotate_range, scale_range, shear_range, prob)` | ranges: `float\|tuple` | Combined affine in one step |
| `with_intensity_jitter(shift, scale, prob)` | `float` each | Additive + multiplicative intensity |
| `with_gaussian_noise(mean, std, prob)` | `float` each | Additive Gaussian noise |
| `with_gaussian_smooth(sigma_range, prob)` | `tuple[float,float]` | Gaussian blur |
| `with_elastic_deformation(magnitude_range, spacing, prob)` | 2D only | Grid-based deformation |
| `with_elastic_deformation(sigma_range, magnitude_range, prob)` | 3D only | Elastic deformation |
| `with_bbox_as_mask(keep_mask)` | `bool` | Convert bbox annotations to a binary mask |
| `with_transpose(indices, keys)` | `indices: list[int]`; `keys: list[str]\|None` | Append spatial transpose (e.g. `[0,2,1]` for 2D, `[0,3,2,1]` for 3D) |
| `with_spacing(pixdim)` | `tuple[float,float,float]` | 3D only — override voxel spacing |
| `add_transform(t)` | any MONAI transform | Append a custom transform |

---

## HU windowing

HU windowing clips CT intensity to a clinically meaningful range before percentile normalisation. Use it only for CT; pass `hu_window=None` for MRI.

| Anatomy | Typical window |
|---------|----------------|
| Lung | `(-1000, 400)` |
| Abdomen | `(-150, 250)` |
| Brain | `(0, 80)` |
| Bone | `(-500, 1500)` |
| Soft tissue | `(-160, 240)` |

---

## dtype control

```python
# For GPU training (Ampere+)
t = RadiologyTransform2D(dtype=torch.bfloat16)

# For CPU or older GPUs
t = RadiologyTransform2D(dtype=torch.float32)
```

The dataset constructor's `dtype` argument is forwarded to the default transform automatically.

---

## Using a custom transform

Pass any MONAI `Compose` or transform directly to the dataset constructor:

```python
import monai.transforms as mn
from radharmony.dataset import CheXpertDataset

my_transform = mn.Compose([
    mn.LoadImageD(keys=["img"]),
    mn.EnsureChannelFirstD(keys=["img"]),
    mn.ResizeD(keys=["img"], spatial_size=(512, 512)),
    mn.ToTensorD(keys=["img"], dtype=torch.float32),
])

ds = CheXpertDataset(base_image_dir="...", transform=my_transform, output_cls=True)
```

---

## Key invariants: 3D orientation + axis order

The 3D pipeline performs two coupled steps before intensity normalisation:

1. **Orientation normalisation.** `_SITKOrientD(axcodes=...)` reorients every volume to a canonical axis code (default `"IPL"`) using SimpleITK's `DICOMOrientImageFilter`. When `output_bbox=True`, `_ReorientBbox` runs **before** the image reorient and pre-permutes bbox coord pairs so they remain aligned with the reoriented image.
2. **Axis transpose.** `TransposeD([0, 3, 2, 1])` then converts ITK's `(W, H, D)` axis order into RadHarmony's `(D, H, W)` layout, which matches the `[dim0_min, dim0_max, dim1_min, dim1_max]` bounding-box convention.

Both steps are gated by `base_transpose=True` and must be kept paired — dropping either one (or changing `axcodes` between the image and bbox reorient) will misalign 3D bbox coordinates. See [Architecture → Pipeline invariants](architecture.md#pipeline-invariants) for details.
