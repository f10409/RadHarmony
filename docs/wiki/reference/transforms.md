# Transforms

<!-- blocks: run=ci -->

RadHarmony has 2 transform builders: `RadiologyTransform2D` and `RadiologyTransform3D`. Each one makes a
MONAI `Compose` with the load and preprocessing steps. The `with_*` methods add augmentation steps.
`get_transform()` returns the `Compose`. A dataset class with `transform=None` uses one of them.

## Default 2-D steps

```
_ParseBboxD → LoadImageD (ITKReader) → EnsureChannelFirstD → drop the DICOM axis →
TransposeD [0, 2, 1] (if base_transpose) → fix the missing high bit → VOI LUT →
set channels (D-42) → ScaleIntensityRangePercentilesD (0.5 and 99.5 to [-1, 1]) →
mask values above 0 to 1 (if "mask") → ResizeD (longest edge) → SpatialPadD (if pad) →
[augmentation] → box to tensor → ToTensorD → SelectItemsD
```

- A 2-D mask in a sample holds 0 and 1, also when the mask file holds 0 and 255 (D-14).
- The pad value is -1 (black) for the image and 0 for the mask.

```python
import torch
from radharmony.dataset.transforms import RadiologyTransform2D

t = RadiologyTransform2D(img_size=224, dtype=torch.float32).get_transform()
print(len(t.transforms))
# 15
```

## Default 3-D steps

```
_ParseBboxD → LoadImageD (ITKReader) → EnsureChannelFirstD →
_ReorientBbox (if base_transpose) → _SITKOrientD (axcodes, if base_transpose) →
SpacingD (if pixdim, moves the box) → TransposeD [0, 3, 2, 1] (if base_transpose) →
ScaleIntensityRangeD (HU clip, if hu_window) → ScaleIntensityRangePercentilesD (0.5 and 99.5 to [-1, 1]) →
ResizeD (longest edge) → SpatialPadD (if pad) → [augmentation] → box to tensor → ToTensorD → SelectItemsD
```

```python
from radharmony.dataset.transforms import RadiologyTransform3D

t = RadiologyTransform3D(
    img_size=112,
    hu_window=(-1000, 400),  # a CT lung window
    pixdim=(1.5, 1.5, 1.5),
    dtype=torch.float32,
).get_transform()
print(len(t.transforms))
# 15
```

## Constructor arguments

### `RadiologyTransform2D`

<!-- args: radharmony.dataset.transforms.RadiologyTransform2D -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `img_size` | `int` | `224` | The size of the longest edge after the resize, in pixels. With `pad=True`, the output is `img_size` × `img_size` |
| `output_keys` | `set[str]` | `None` | The keys to keep in the sample. `None` gives `{"img", "cls"}`. The default transform of a dataset class sets the keys of its flags. A transform of your own must list them, or the dataset class warns (D-11) |
| `pad` | `bool` | `True` | After the resize, pad to a square with -1 (black). A mask gets 0. `False` keeps the aspect ratio |
| `base_transpose` | `bool` | `True` | Add the `[0, 2, 1]` transpose after the load. It corrects the axis order of the ITK reader |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The type of `img` and `bbox_mask`. `cls`, `reg`, `mask` and `bbox` are always float32 |
| `channels` | `int` | `1` | The channel count of `img`: 1 or 3 (D-42). With 1, an RGB or RGBA file becomes gray. With 3, a gray file repeats to 3 channels |

### `RadiologyTransform3D`

<!-- args: radharmony.dataset.transforms.RadiologyTransform3D -->
| Argument | Type | Default | Description |
|---|---|---|---|
| `img_size` | `int` | `112` | The size of the longest edge after the resize, in voxels. With `pad=True`, the output is a cube |
| `output_keys` | `set[str]` | `None` | The keys to keep in the sample. `None` gives `{"img", "cls"}` |
| `hu_window` | `tuple[float, float]` | `None` | The HU clip range, for example `(-1000, 400)`. `None` gives no clip. Use `None` for MRI |
| `pad` | `bool` | `True` | After the resize, pad to a cube with -1 (black). A mask gets 0 |
| `base_transpose` | `bool` | `True` | Add the orientation step and the `[0, 3, 2, 1]` transpose. The boxes need both ([3-D orientation](#3-d-orientation-and-axis-order)) |
| `pixdim` | `tuple[float, float, float]` | `None` | The voxel spacing to resample to, in mm. `None` gives no resample |
| `axcodes` | `str` | `"IPL"` | The orientation of the volume, for example `"RAS"`, `"LPS"` or `"IPL"`. `_SITKOrientD` and `_ReorientBbox` both use it |
| `dtype` | `torch.dtype` | `torch.bfloat16` | The type of `img` and `bbox_mask`. `cls`, `reg`, `mask` and `bbox` are always float32 |

## Augmentation methods

Call the `with_*` methods before `get_transform()`. Each method returns the builder, so you can chain them.
The steps run in the order of the calls, after the preprocessing steps.

```python
t = (
    RadiologyTransform2D(img_size=224, dtype=torch.float32)
    .with_flip(spatial_axis=0, prob=0.5)
    .with_affine(translate_range=(10.0, 10.0), rotate_range=(0.175,), prob=0.5)
    .with_intensity_jitter(shift=0.1, scale=0.1, prob=0.5)
    .with_gaussian_noise(mean=0.0, std=0.05, prob=0.5)
    .get_transform()
)
print(len(t.transforms))
# 20
```

| Method | Defaults | Notes |
|---|---|---|
| `with_flip(spatial_axis, prob)` | `0`, `0.5` | A random flip. In 2-D, axis 0 is up-down and axis 1 is left-right |
| `with_affine(translate_range, rotate_range, scale_range, shear_range, prob)` | `(10.0, 10.0)`, `(0.175,)`, `(0.1, 0.1)`, `(0.05,)`, `0.5` | One random affine step. The translation is in pixels. The rotation is in radians (0.175 is about 10 degrees) |
| `with_intensity_jitter(shift, scale, prob)` | `0.1`, `0.1`, `0.5` | A random shift and a random scale of the values. Each one runs with `prob` |
| `with_gaussian_noise(mean, std, prob)` | `0.0`, `0.05`, `0.5` | Adds random Gaussian noise |
| `with_gaussian_smooth(sigma_range, prob)` | `(0.5, 1.5)`, `0.2` | A random Gaussian blur |
| `with_elastic_deformation(magnitude_range, spacing, prob)` | `(50.0, 150.0)`, `20`, `0.2` | 2-D only. A random grid warp. `spacing` is the grid step in pixels |
| `with_elastic_deformation(sigma_range, magnitude_range, prob)` | `(5.0, 7.0)`, `(50.0, 150.0)`, `0.2` | 3-D only. A random elastic warp |
| `with_bbox_as_mask(keep_mask)` | `False` | With `keep_mask=True`, the sample also has `bbox_mask`, one channel for each box. The box also moves without this call |
| `with_transpose(indices, keys)` | no default, `None` | Adds a transpose, for example `[0, 2, 1]` for 2-D. `None` gives `img`, and `mask` when it is in `output_keys` |
| `with_spacing(pixdim)` | `(1.0, 1.0, 1.0)` | 3-D only. Sets `pixdim` |
| `add_transform(t)` | | Adds any MONAI transform. It does not move the box |

`with_affine` and `with_elastic_deformation` fill the empty border with black (D-37). For
`RadiologyTransform2D` and `RadiologyTransform3D`, this is -1. For `RadiologyEncoderTransform`, it is the
darkest value of each channel of the image, because the encoder has already normalized the image.

## Boxes

When `"bbox"` is in `output_keys`, the box in the sample always matches the returned image (D-28). Each
step moves the box in the same way as the image:

| Step | How the box moves |
|---|---|
| Resize | No change. A box is a fraction of the image size, and the resize stretches the whole image |
| `with_spacing` or `pixdim` (3-D resample) | Exact math. MONAI keeps the first voxel center and rounds the output size, so the output covers a different area than the input. Each box corner goes through the voxel map of the input and output affines. Then the box is clipped to [0, 1] |
| Pad (`pad=True`) | Exact math. Example: a 200 × 100 (H × W) image becomes 64 × 32, then 64 × 64 with 16 pixels on the left and right. `dim1_min = 0.0` becomes `0.25` |
| `with_flip` | Exact math: `[lo, hi]` becomes `[1 - hi, 1 - lo]` on each flipped axis |
| `with_transpose` | Exact math: the coordinate pairs change places as the axes do |
| `with_affine`, `with_elastic_deformation` | Mask round trip. Each box gets its own mask channel, and the warp moves the masks with the image. The new box is the box around the warped mask. The round trip moves an edge by 0.5 pixel or less. The step drops a box that leaves the image, with its label |
| `add_transform` | No change. A custom step does not move the box |

- In the sample, `bbox` is a float32 tensor of shape `(N, 4)` for 2-D or `(N, 6)` for 3-D, also when
  `img` is bfloat16. An image with no box gives shape `(0, 4)` or `(0, 6)`.
- `bbox_labels` and `bbox_findings` are lists with one entry for each box.
- The table can keep the boxes as a list, as a JSON string, or as the Python list text that `to_csv`
  writes (for example `"['nodule', 'mass']"`). The transform reads all 3.
- `RadiologyEncoderTransform` does not move the box with the crop or pad of the encoder processor (a known
  gap of D-28). It warns when `"bbox"` is in `output_keys` and the processor does more than resize.
- To batch samples with a different number of boxes, use `bbox_collate`
  ([Data dict keys](api-datasets.md#data-dict-keys)).

## HU windows

`hu_window` clips the CT values before the percentile scaling. Use it only for CT. For MRI, use
`hu_window=None`. Common windows:

| Body part | Window |
|---|---|
| Lung | `(-1000, 400)` |
| Abdomen | `(-150, 250)` |
| Brain | `(0, 80)` |
| Bone | `(-500, 1500)` |
| Soft tissue | `(-160, 240)` |

RAD-ChestCT and MedSeg COVID-19 CT use the lung window `(-1000, 400)`.

## dtype

`dtype` sets the type of `img`. `torch.bfloat16` (the default) is for a GPU that supports it. Use
`torch.float32` on a CPU or an older GPU. The `dtype` argument of a dataset class goes to its default
transform.

## A transform of your own

A dataset class takes any MONAI transform as `transform`. The transform gets a dict with the path of the
image in `img`. It must load the image. It has none of the default steps, so it does not scale the values
and does not move a box.

<!-- block: run=local data=montgomery_cxr -->
```python
import monai.transforms as mn
from radharmony.dataset import MontgomeryCXRDataset

my_transform = mn.Compose([
    mn.LoadImageD(keys=["img"]),
    mn.EnsureChannelFirstD(keys=["img"]),
    mn.ResizeD(keys=["img"], spatial_size=(512, 512)),
    mn.ToTensorD(keys=["img", "cls"], dtype=torch.float32, track_meta=False),
    mn.SelectItemsD(keys=["img", "cls"]),
])
ds = MontgomeryCXRDataset(
    base_image_dir="/data/MontgomerySet/CXR_png/", transform=my_transform, output_cls=True, cache_dir=None
)
sample = ds.get_datasets()[0]
print(sample["img"].shape, sample["img"].max().item())
# torch.Size([1, 512, 512]) 255.0
```

To keep the default steps and add a step of your own, use `add_transform()`.

## 3-D orientation and axis order

The 3-D transform has 2 linked steps before the value scaling (D-12):

1. Orientation. `_SITKOrientD(axcodes=...)` turns each volume to one axis code (default `"IPL"`) with the
   SimpleITK `DICOMOrientImageFilter`. `_ReorientBbox` runs before it and changes the order of the box
   coordinate pairs, so the box stays on the turned image.
2. Axis transpose. `TransposeD([0, 3, 2, 1])` changes the ITK axis order `(W, H, D)` to the RadHarmony
   order `(D, H, W)`. This order matches the box order `[dim0_min, dim0_max, dim1_min, dim1_max, ...]`.

Keep both steps. `base_transpose=True` turns on both. If you remove one step, or use another `axcodes` for
the image and the box, the 3-D boxes do not match the image.
