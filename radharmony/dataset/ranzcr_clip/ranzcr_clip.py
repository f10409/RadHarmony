"""MONAI dataset for the RANZCR CLiP catheter & line classification dataset.

Wraps :class:`RANZCRClipHarmonizer`.  Multi-label classification (11 binary
targets) is the primary task and works out of the box with ``output_cls=True``.

Mask outputs are optional.  ``train_annotations.csv`` has polyline
annotations that trace each catheter / line for some training images.
The harmonizer :class:`RANZCRClipHarmonizer` aggregates and JSON-encodes them.
The method :meth:`RANZCRClipHarmonizer._decode_mask` rasterizes them via
``cv2.polylines`` with a configurable ``mask_line_thickness``.  Pre-decode
to PNG files via ``mask_output_dir`` (mirrors the SIIM-ACR-PTX flow, MONAI
cannot read raw JSON strings as image paths).

By default, the harmonized DataFrame includes both train and test rows
distinguished by a ``split`` column ("train" or "test").  Test rows have NaN
labels (no public labels for the competition holdout). Call
:meth:`get_datasets_predefined` to obtain split-specific datasets.

This mixes splits: ``get_datasets()`` puts unlabeled test rows into train
and validation. It is not for benchmarks. Use :class:`RANZCRClipTrainDataset`
(labeled rows) and :class:`RANZCRClipTestDataset` (unlabeled test images).
"""

import os
import warnings

import monai as mn
import pandas as pd
import torch

from radharmony.harmonizer import RANZCRClipHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.data_utils import get_data_dict
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset, _has_random_step
from ..transforms import RadiologyTransform2D


class _RequireLabels(mn.transforms.Transform):
    """Raise a clear error when a sample has NaN labels (an unlabeled Kaggle test image).

    It is a deterministic MONAI transform, so PersistentDataset still caches
    the steps after it.
    """

    def __call__(self, d: dict) -> dict:
        if any(pd.isna(v) for v in d.get("cls", [])):
            raise ValueError(
                f"{d['img']} is a RANZCR CLiP test image with no labels, so output_cls=True cannot "
                "load it. Use RANZCRClipTrainDataset to train and RANZCRClipTestDataset for the "
                "test images, or pass include_test_split=False."
            )
        return d


@register_dataset("ranzcr_clip")
class RANZCRClipDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for RANZCR CLiP.

    Args:
        base_image_dir: Root of the Kaggle download
            (for example ``~/.cache/kagglehub/competitions/ranzcr-clip-catheter-line-classification/``).
            Contains ``train/``, ``test/``, and the CSVs.
        csv_path: Path to ``train.csv``.  Auto-inferred when ``None``.
        annotations_csv_path: Path to ``train_annotations.csv``.  Auto-inferred
            when ``None``.  Required only when ``output_mask=True``.
        mask_output_dir: Directory where rasterized polyline PNGs are written.
            Required when ``output_mask=True`` (polyline JSON cannot be loaded
            directly by MONAI's image loader).
        mask_num_cores: Worker threads for mask pre-decoding.
        mask_line_thickness: Pixel width passed to ``cv2.polylines`` when
            rasterizing polylines.  Default 15 mirrors the official challenge
            starter notebook.
        include_test_split: When ``True`` (default), the harmonized DataFrame
            includes test/<id>.jpg rows with NaN labels and ``split="test"``.
            With ``output_cls=True``, a sample of such a row raises a
            ``ValueError`` when it loads.
        transform: MONAI Compose transform.  Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Yield rasterized polyline masks under key ``mask``.
            Requires ``mask_output_dir`` and ``annotations_csv_path``.
        output_report / output_bbox / output_reg: Not supported. A warning, and no key.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask"})

    #: The train source files must exist (D-35). A test-split subclass that reads
    #: only its test files sets False.
    _TRAIN_FILES_REQUIRED: bool = True

    LABEL_COLS = [
        "cvc_abnormal",
        "cvc_borderline",
        "cvc_normal",
        "ett_abnormal",
        "ett_borderline",
        "ett_normal",
        "ngt_abnormal",
        "ngt_borderline",
        "ngt_incompletely_imaged",
        "ngt_normal",
        "swan_ganz_catheter_present",
    ]
    _HARMONIZER_CLS = RANZCRClipHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        annotations_csv_path: str = None,
        mask_output_dir: str = None,
        mask_num_cores: int = 1,
        mask_line_thickness: int = 15,
        include_test_split: bool = True,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if csv_path:
            csv_path = os.path.expanduser(csv_path)
        if annotations_csv_path:
            annotations_csv_path = os.path.expanduser(annotations_csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_image_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        if output_report:
            warnings.warn(
                "RANZCR CLiP does not provide radiology reports. "
                "output_report will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            output_report = False
        if output_bbox:
            warnings.warn(
                "RANZCR CLiP does not provide bounding boxes (annotations are "
                "polylines, not regions). output_bbox will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            output_bbox = False

        if (
            output_mask
            and mask_output_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "output_mask=True requires mask_output_dir to be set. "
                "RANZCR CLiP polyline annotations are stored as JSON strings "
                "and must be rasterized to PNG files before MONAI can load them.\n"
                "Pass a directory:\n"
                "    RANZCRClipDataset(..., output_mask=True, mask_output_dir='/path/to/masks')"
            )

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = (
                infer_path(
                    base_image_dir, "train.csv", user_path=csv_path or "",
                    required=self._TRAIN_FILES_REQUIRED, arg="csv_path",
                )
                or csv_path
            )
            # The masks come from train_annotations.csv, so output_mask needs it.
            self._annotations_csv_path = (
                infer_path(
                    base_image_dir,
                    "train_annotations.csv",
                    user_path=annotations_csv_path or "",
                    required=output_mask,
                    arg="annotations_csv_path",
                )
                or annotations_csv_path
            )
        else:
            self._csv_path = csv_path
            self._annotations_csv_path = annotations_csv_path

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_mask:
                output_keys.add("mask")
            transform = RadiologyTransform2D(
                img_size=224,
                output_keys=output_keys,
                dtype=dtype,
            ).get_transform()

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

        self._mask_output_dir = mask_output_dir
        self._mask_num_cores = mask_num_cores
        self._mask_line_thickness = mask_line_thickness
        self._include_test_split = include_test_split

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = RANZCRClipHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            annotations_csv_path=self._annotations_csv_path,
            mask_line_thickness=self._mask_line_thickness,
            include_test_split=self._include_test_split,
        )
        return harmonizer.harmonize(
            mask_output_dir=self._mask_output_dir,
            mask_num_cores=self._mask_num_cores,
        )

    def _make_dataset(self, dicts: list, cache_subdir: str, transform=None):
        """Build the dataset. With ``output_cls``, check the labels before the transform."""
        if self.output_cls:
            transform = transform if transform is not None else self.transform
            # A flat list: a Compose inside a Compose is one random step to the cache.
            steps = list(transform.transforms) if isinstance(transform, mn.transforms.Compose) else (
                [transform] if transform else [])
            transform = mn.transforms.Compose([_RequireLabels()] + steps)
        return super()._make_dataset(dicts, cache_subdir, transform=transform)

    def get_datasets_predefined(
        self,
        num_cores: int = 2,
        train_transform=None,
        test_transform=None,
    ):
        """Return ``(train, test)`` per the dataset's official train/test split.

        Slices the harmonized DataFrame on its ``split`` column.  Test rows
        have NaN labels and are intended for inference / submission only.
        With ``test_transform=None``, the test split uses the dataset transform.
        A random step in it gives a warning (D-49).

        Raises:
            ValueError: If the harmonized DataFrame has no ``split`` column
                (for example ``include_test_split=False`` was set), or either split
                is empty.
        """
        df = self._get_harmonized_df()
        if "split" not in df.columns:
            raise ValueError(
                "Harmonized DataFrame has no 'split' column; pass "
                "include_test_split=True to expose the official competition splits."
            )
        self.LABEL_COLS = sorted(self.LABEL_COLS)
        cols = self._cols

        out = []
        for split_name, transform in (("train", train_transform), ("test", test_transform)):
            sub = df[df["split"] == split_name]
            if sub.empty:
                raise ValueError(f"Split '{split_name}' is empty in the harmonized DataFrame.")
            if transform is None and split_name != "train" and _has_random_step(self.transform):
                warnings.warn(
                    f"The dataset transform has a random step, and the {split_name} split uses it, because "
                    f"{split_name}_transform is None. So the {split_name} images get augmentation. "
                    f"Pass {split_name}_transform= with no random step (D-49).",
                    stacklevel=2,
                )
            dicts = get_data_dict(sub, self.base_image_dir, "image_path", cols, num_cores)
            out.append(self._make_dataset(dicts, cache_subdir=split_name, transform=transform))
        return tuple(out)
