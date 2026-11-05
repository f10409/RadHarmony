# The dataset class

The dataset class is a PyTorch `Dataset` on top of the harmonized table. Read D-09, D-11, D-14 and D-35 in
`docs/wiki/design.md` first.

## Files

- `radharmony/dataset/<name>/<name>.py`: the class.
- `radharmony/dataset/<name>/__init__.py`: the import, and `__all__`. Example for `my_dataset`:

  ```python
  from .my_dataset import MyDatasetDataset

  __all__ = ["MyDatasetDataset"]
  ```

Inside `<name>.py`, import the base files with `..`, for example
`from ..base import BaseRadiologicalDataset` and `from ..transforms import RadiologyTransform2D`.

## Template

The class for the harmonizer template in `harmonizer.md`. The template runs on fake files,
with `output_cls=True`, with an unsupported flag, and with `harmonizer_path`. It imports the harmonizer from
`radharmony.harmonizer`, so export the harmonizer first (SKILL.md step 6.2).

```python
"""Dataset class for My Dataset."""

import torch

from radharmony.harmonizer import MyDatasetHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path

from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("my_dataset")
class MyDatasetDataset(BaseRadiologicalDataset):
    """PyTorch dataset for My Dataset.

    Args:
        base_image_dir: The ``images/`` folder, for example ``/data/my_dataset/images/``.
        csv_path: ``metadata.csv``. Without it, the class looks in ``base_image_dir`` and its parent (D-35).
        label_csv_path: ``labels.csv``. The class needs it only with ``output_cls=True``.
        transform: The image steps. ``None`` gives ``RadiologyTransform2D``.
        cache_dir: The MONAI cache folder. ``None`` gives no cache.
        output_cls: Add the ``cls`` tensor to the sample.
        output_mask: Not supported. The class warns and adds no key (D-11).
        output_report: Not supported. The class warns and adds no key (D-11).
        output_bbox: Not supported. The class warns and adds no key (D-11).
        output_reg: Not supported. The class warns and adds no key (D-11).
        harmonized_df: A harmonized table to use.
        harmonizer: A harmonizer to use.
        harmonizer_path: A saved harmonizer.
        dtype: The type of ``img``. The labels stay float32.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})
    LABEL_COLS = ["cardiomegaly", "pleural_effusion"]  # snake_case and sorted: the order of cls (D-05)
    _HARMONIZER_CLS = MyDatasetHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        output_reg: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        preset = harmonized_df is not None or harmonizer is not None or harmonizer_path is not None
        if harmonizer_path is not None and base_image_dir is None:
            base_image_dir = self._HARMONIZER_CLS.load_from_saved(harmonizer_path).base_image_dir
        if base_image_dir is None and not preset:
            raise ValueError("base_image_dir is required: give the images/ folder of My Dataset.")
        if not preset:
            # Look only in base_image_dir and its parent (D-35). List every known file name.
            csv_path = infer_path(base_image_dir, "metadata.csv", "metadata.csv.gz",
                                  user_path=csv_path or "", required=True, arg="csv_path")
            label_csv_path = infer_path(base_image_dir, "labels.csv", user_path=label_csv_path or "",
                                        required=output_cls, arg="label_csv_path") or None
        self._csv_path, self._label_csv_path = csv_path, label_csv_path

        if transform is None:
            flags = {"cls": output_cls, "mask": output_mask, "bbox": output_bbox,
                     "report": output_report, "reg": output_reg}
            # Keep only the supported keys. A missing key makes the transform raise (D-11).
            keys = {"img"} | {k for k, on in flags.items() if on and k in self.SUPPORTED_OUTPUTS}
            if "bbox" in keys:
                keys.add("bbox_labels")
            transform = RadiologyTransform2D(img_size=224, output_keys=keys, dtype=dtype).get_transform()

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_bbox=output_bbox,
            output_reg=output_reg,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = MyDatasetHarmonizer(
            csv_path=self._csv_path,
            label_csv_path=self._label_csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
```

## Rules

1. **`LABEL_COLS`** holds the snake_case names of the harmonized table, sorted with `sorted()`. The `cls`
   vector uses the order `sorted(ds.LABEL_COLS)` (`radharmony/dataset/base.py`, `_cols`). So
   `LABEL_COLS[j]` names `cls[j]` only when the list is sorted. `tests/test_label_order.py` checks it
   (D-05).
2. **`_HARMONIZER_CLS`** names the harmonizer. `harmonizer_path=` needs it.
3. **`_get_harmonized_df()`** calls `_try_resolve_preset_harmonized()` first. Then `harmonized_df`,
   `harmonizer` and `harmonizer_path` work.
4. **`super().__init__()`** gets `harmonized_df`, `harmonizer`, `harmonizer_path` and all 5 output flags.
5. **The output flags** (D-11). The class takes `output_cls`, `output_mask`, `output_report`, `output_bbox`
   and `output_reg` (`tests/test_output_flags.py`). `SUPPORTED_OUTPUTS` lists the outputs that the class
   gives. The base class warns for a flag that is not in it. Do not add your own warning. The default
   transform keeps only the supported keys, because a missing key makes the transform raise.
6. **`infer_path()`** finds a source file in `base_image_dir` and its parent folder only (D-35). Give every
   known file name (the `filename_variants` of the config). A file that the class always reads uses
   `required=True`. A file that only one output reads uses the flag of that output, for example
   `required=output_cls`.
7. **The default transform.** Use `RadiologyTransform2D` for 2-D and `RadiologyTransform3D` for 3-D. A 3-D
   CT dataset gives `hu_window`. An MRI dataset leaves `hu_window=None`. A dataset of color images passes
   `channels=3`, and a VQA dataset sets `IMG_CHANNELS = 3` (D-42).
8. **`dtype`** (default `torch.bfloat16`) goes to the default transform. It sets the type of `img`. It does
   nothing when the user gives `transform=`.
9. **Official splits** (D-09). Each split is its own class with its own `@register_dataset("<key>")`. A
   class takes no `split=` argument. A split with no labels has no `cls` (D-25).
10. **Masks that need a step first.** When the table holds RLE strings or files to join, add a
    `ValueError` for `output_mask=True` with no `mask_output_dir`. See "RLE masks" in `data_traps.md`.
11. **A folder as `image_path`** (a DICOM series). Override `verify_images` here and in the harmonizer. See
    "DICOM series folders" in `data_traps.md`.
