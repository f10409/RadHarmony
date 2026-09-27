"""Base radiological dataset wrapping MONAI PersistentDataset."""

import warnings

import pandas as pd
import torch
import monai as mn
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from radharmony.utils.data_utils import get_data_dict, split_data, kfold_splits


# Columns not treated as classification labels when inferring labels from a raw DataFrame
_HARMONIZED_NON_LABEL_COLS = frozenset(
    {
        "patient_id",
        "study_id",
        "series_id",
        "image_path",
        "view_position",
        "mask_path",
        "bbox",
        "report",
        "split",
    }
)


def _fix_missing_highbit(x):
    """Re-read pixels with pydicom for DICOMs that have no HighBit tag.

    ITKReader (GDCM) cannot decode these and returns a constant image.
    pydicom assumes HighBit = BitsStored - 1. The result is written in the
    same form ITK uses (rescale applied, MONOCHROME1 as s*(M - raw) + b), so
    later steps are unchanged. No-op for non-DICOM images or files with HighBit.
    Module-level so it is picklable for DataLoader workers.
    """
    path = str(x.meta.get("filename_or_obj", ""))
    if not path.lower().endswith((".dcm", ".dicom")):
        return x
    if "HighBit" in pydicom.dcmread(path, stop_before_pixels=True):
        return x
    ds = pydicom.dcmread(path)
    try:
        arr = ds.pixel_array.astype("float32")
    except Exception as e:  # e.g. compressed data with no pydicom decoder
        warnings.warn(f"No HighBit and pydicom cannot decode {path}: {e}")
        return x
    s = float(ds.get("RescaleSlope", 1) or 1)
    b = float(ds.get("RescaleIntercept", 0) or 0)
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = (2 ** int(ds.BitsStored) - 1) - arr
    arr = s * arr + b
    if arr.shape != tuple(x.shape[1:]):
        arr = arr.T
    x[:] = torch.from_numpy(arr).unsqueeze(0)
    return x


def _apply_voi_lut(x):
    """Apply DICOM VOI LUT (WindowCenter/Width or LUT sequence) to a MetaTensor.

    No-op for non-DICOM images. Reads only the DICOM header (no pixel re-read).
    Module-level so it is picklable for DataLoader workers.
    """
    path = str(x.meta.get("filename_or_obj", ""))
    if not path.lower().endswith(".dcm"):
        return x
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    arr = x.numpy().squeeze()
    # ITKReader (GDCM) loads MONOCHROME1 as s*(M - raw) + b (flip, then
    # rescale), but the window tags refer to s*raw + b: undo the flip,
    # window, then flip back.
    flip = getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1" and (
        "WindowCenter" in ds or "VOILUTSequence" in ds
    )
    if flip:
        s = float(ds.get("RescaleSlope", 1) or 1)
        b = float(ds.get("RescaleIntercept", 0) or 0)
        # M is 2**BitsStored - 1, except JPEG 2000 where GDCM uses
        # 2**BitsAllocated - 1; flipped values above the first rule it out.
        m = 2 ** int(ds.BitsStored) - 1
        if (arr.max() - b) / s > m:
            m = 2 ** int(ds.BitsAllocated) - 1
        arr = (s * m + 2 * b) - arr
    arr = apply_voi_lut(arr, ds).astype("float32")
    if flip:
        arr = arr.max() + arr.min() - arr
    x[:] = torch.from_numpy(arr).unsqueeze(0)
    return x


def _convert_monochrome1_to_2(x):
    """Invert pixel values if the DICOM PhotometricInterpretation is MONOCHROME1.

    MONOCHROME1 encodes air as white (high values) and bone as black (low
    values), opposite to the MONOCHROME2 convention used by most models.
    This inverts the image so that all inputs follow the MONOCHROME2 convention.

    No-op for non-DICOM images or images already in MONOCHROME2.
    Module-level so it is picklable for DataLoader workers.
    """
    path = str(x.meta.get("filename_or_obj", ""))
    # print(f"[monochrome-debug] filename_or_obj={path!r}, endswith_dcm={path.lower().endswith('.dcm')}")
    if not path.lower().endswith(".dcm"):
        return x
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    pi = getattr(ds, "PhotometricInterpretation", "")
    # print(f"[monochrome-debug] PhotometricInterpretation={pi!r}, will_invert={pi == 'MONOCHROME1'}")
    if pi == "MONOCHROME1":
        x[:] = x.max() - x
    return x


def _bbox_rect_for_projection(b, spatial, projection_axis):
    """Return (xy, width, height) for a matplotlib Rectangle, or None.

    Args:
        b: Normalized bbox as ``[dim0_min, dim0_max, dim1_min, dim1_max, ...]``
            with values in ``[0, 1]``.  4 values for 2-D, 6 for 3-D.
        spatial: Spatial shape of the image ``(H, W)`` or ``(D, H, W)``.
        projection_axis: Axis collapsed when projecting 3-D → 2-D.

    Returns:
        ``((x, y), width, height)`` in pixel coords, or ``None`` if bbox
        format is unrecognised.
    """
    if len(spatial) == 2 and len(b) == 4:
        H, W = spatial
        y_min, y_max = b[0] * H, b[1] * H
        x_min, x_max = b[2] * W, b[3] * W
    elif len(spatial) == 3 and len(b) == 6:
        remaining = [i for i in range(3) if i != projection_axis]
        row_ax, col_ax = remaining[0], remaining[1]
        y_min = b[row_ax * 2] * spatial[row_ax]
        y_max = b[row_ax * 2 + 1] * spatial[row_ax]
        x_min = b[col_ax * 2] * spatial[col_ax]
        x_max = b[col_ax * 2 + 1] * spatial[col_ax]
    else:
        return None
    return (x_min, y_min), x_max - x_min, y_max - y_min


class BaseRadiologicalDataset:
    """Base class for MONAI-backed radiological datasets.

    Subclasses must define :attr:`LABEL_COLS` and implement
    :meth:`_get_harmonized_df`.

    Args:
        base_image_dir: Root directory prepended to relative image paths.
        transform: MONAI transform pipeline applied to each sample.
        cache_dir: Root directory for MONAI PersistentDataset cache.
            Pass ``None`` to use an in-memory ``Dataset`` (no caching).
        output_cls: Include label vector (``cls``) in each sample.
        output_mask: Include segmentation mask (``mask``) in each sample.
            Requires ``mask_path`` to be populated in the harmonized DataFrame
            (e.g. via ``mask_output_dir``).
        output_report: Include radiology report path (``report``) in each sample.
            Requires a ``report_path`` column in the harmonized DataFrame.
        output_bbox: Include bounding box (``bbox``) in each sample.
            Requires a ``bbox`` column in the harmonized DataFrame.
        harmonized_df: If set, skip harmonization and use this DataFrame
            (same columns as :meth:`harmonize` would produce).
        harmonizer: Use this harmonizer's :attr:`~radharmony.harmonizer.base.BaseHarmonizer.harmonized_df`
            (call :meth:`~radharmony.harmonizer.base.BaseHarmonizer.harmonize` first, assign
            ``harmonized_df``, or :meth:`~radharmony.harmonizer.base.BaseHarmonizer.load_from_saved`).
        harmonizer_path: Load a harmonizer from a pickle saved via
            :meth:`radharmony.harmonizer.base.BaseHarmonizer.save`.  Concrete dataset
            classes must set :attr:`_HARMONIZER_CLS` to the matching harmonizer type.

    Resolution order for the harmonized table:
    ``harmonized_df`` → ``harmonizer_path`` → ``harmonizer`` → subclass implementation.
    """

    LABEL_COLS: list = []
    #: Continuous target columns for regression tasks.  When a subclass sets
    #: this and the user passes ``output_reg=True``, each sample gets a
    #: ``"reg"`` key aggregating those columns into a list (mirror of how
    #: ``cls`` aggregates ``LABEL_COLS``).  Empty by default — only regression
    #: datasets populate it (e.g. RSNA Bone Age).
    REG_COLS: list = []
    _HARMONIZER_CLS = None
    #: Output keys supported by this dataset.  Concrete subclasses override
    #: this to advertise which output flags will actually populate data.
    #: Passing an unsupported flag (e.g. ``output_mask=True`` on a cls-only
    #: dataset) emits a ``UserWarning`` from ``__init__``.
    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask", "bbox", "report", "reg"})

    #: Whether this dataset carries a per-box ``bbox_findings`` column (parallel
    #: to ``bbox`` / ``bbox_labels``).  When True and ``output_bbox`` is set, the
    #: findings list is threaded into each sample alongside the boxes.  Left
    #: False for datasets whose harmonized table has no such column.
    SUPPORTS_BBOX_FINDINGS: bool = False

    # Maps dataset output key → harmonized DataFrame column
    _EXTRA_COL_MAP: dict = {
        "mask": "mask_path",
        "report": "report",
        "bbox": "bbox",
        "bbox_labels": "bbox_labels",
        "bbox_findings": "bbox_findings",
    }

    def __init__(
        self,
        base_image_dir: str,
        transform,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        output_reg: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
    ):
        self.base_image_dir = base_image_dir
        self.transform = transform
        self.cache_dir = cache_dir
        self.output_cls = output_cls
        self.output_mask = output_mask
        self.output_report = output_report
        self.output_bbox = output_bbox
        self.output_reg = output_reg
        self._preset_harmonized_df = harmonized_df
        self._user_harmonizer = harmonizer
        self._harmonizer_save_path = harmonizer_path

        # Warn for flags that this dataset class does not support.
        _flag_map = {
            "output_cls": "cls",
            "output_mask": "mask",
            "output_report": "report",
            "output_bbox": "bbox",
            "output_reg": "reg",
        }
        for _flag, _key in _flag_map.items():
            if getattr(self, _flag) and _key not in self.SUPPORTED_OUTPUTS:
                warnings.warn(
                    f"{_flag}=True is not supported by {type(self).__name__}; "
                    f"the '{_key}' key will be absent from samples.",
                    UserWarning,
                    stacklevel=3,
                )

    @property
    def _output_keys(self) -> set:
        keys = {"img"}
        if self.output_cls:
            keys.add("cls")
        if self.output_mask:
            keys.add("mask")
        if self.output_report:
            keys.add("report")
        if self.output_bbox:
            keys.add("bbox")
            keys.add("bbox_labels")
            if self.SUPPORTS_BBOX_FINDINGS:
                keys.add("bbox_findings")
        if self.output_reg:
            keys.add("reg")
        return keys

    @property
    def _cols(self) -> dict:
        """Build the unified cols dict for get_data_dict."""
        result = {}
        if self.output_cls:
            result["cls"] = sorted(self.LABEL_COLS)
        if self.output_reg:
            result["reg"] = sorted(self.REG_COLS)
        flags = {
            "mask": self.output_mask,
            "report": self.output_report,
            "bbox": self.output_bbox,
            "bbox_labels": self.output_bbox,
            "bbox_findings": self.output_bbox and self.SUPPORTS_BBOX_FINDINGS,
        }
        result.update(
            {k: self._EXTRA_COL_MAP[k] for k, enabled in flags.items() if enabled}
        )
        return result

    def _make_dataset(self, dicts: list, cache_subdir: str, transform=None):
        t = transform if transform is not None else self.transform
        if self.cache_dir is None:
            return mn.data.Dataset(data=dicts, transform=t)
        return mn.data.PersistentDataset(
            data=dicts,
            transform=t,
            cache_dir=(
                f"{self.cache_dir}/{cache_subdir}" if cache_subdir else self.cache_dir
            ),
        )

    def pre_cache(self, dataset, num_workers: int = 2):
        """Pre-populate the PersistentDataset cache by iterating all items.

        Call this after get_datasets() or get_folds() to warm the cache
        before training begins.

        Args:
            dataset: A PersistentDataset (or plain Dataset) to pre-cache.
            num_workers: Workers for parallel caching.
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        loader = DataLoader(
            dataset, batch_size=1, num_workers=num_workers, collate_fn=lambda x: x
        )
        for _ in tqdm(loader, desc="Pre-caching", total=len(dataset)):
            pass

    def visualize_samples(
        self,
        dataset,
        n: int = 4,
        indices=None,
        per_label: bool = False,
        random_state: int = None,
        projection_axis: int = 0,
    ):
        """Yield one matplotlib Figure per sample showing all active outputs.

        Subplots are added dynamically based on which output keys are present
        in the sample (``img`` always, plus ``mask``, ``cls``, ``bbox``,
        ``report`` when enabled).  3-D volumes are collapsed to a 2-D mean
        projection along *projection_axis* for display.

        Args:
            dataset: A dataset returned by :meth:`get_datasets` or
                :meth:`get_folds`.
            n: Number of samples to randomly visualize when *indices* is
                ``None`` and *per_label* is ``False``.
            indices: Explicit sequence of integer indices into *dataset*.
                When provided, *n* and *per_label* are ignored.
            per_label: When ``True`` and ``LABEL_COLS`` is non-empty, pick
                one random positive example per label by scanning the
                pre-transform data dicts (no image loading required for
                the scan).  Yields one figure per label; labels with no
                positive example in *dataset* are skipped.
            random_state: Seed for reproducible random sampling.
            projection_axis: Spatial axis used for the mean projection when
                collapsing 3-D volumes to 2-D for display.  ``0`` → axial
                (default), ``1`` → coronal, ``2`` → sagittal.

        Yields:
            ``matplotlib.figure.Figure`` — one figure per sample. The caller
            is responsible for displaying (``plt.show()`` / ``display(fig)``)
            and closing (``plt.close(fig)``) each figure.

        Example::

            for fig in ds_obj.visualize_samples(train_ds, n=6):
                display(fig)
                plt.close(fig)

            # one representative image per label
            for fig in ds_obj.visualize_samples(train_ds, per_label=True):
                display(fig)
                plt.close(fig)
        """
        import random
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        rng = random.Random(random_state)

        if indices is not None:
            selected = list(indices)
            per_label = False  # treat explicit indices as normal mode
        elif per_label and self.LABEL_COLS and hasattr(dataset, "data"):
            # Scan pre-transform dicts to find a positive example per label.
            # dataset.data[i]["cls"] is a list/tensor of label values.
            selected = []
            for label_idx, label_name in enumerate(self.LABEL_COLS):
                positives = [
                    i
                    for i, d in enumerate(dataset.data)
                    if float(d["cls"][label_idx]) > 0.5
                ]
                if positives:
                    selected.append((rng.choice(positives), label_name))
                # else: no positive example for this label in the split — skip
        else:
            per_label = False
            pool = list(range(len(dataset)))
            selected = rng.sample(pool, min(n, len(pool)))

        # ── per_label: one grid figure containing all labels ──────────────────
        if per_label:
            import math

            n_labels = len(selected)
            ncols = min(5, n_labels)
            nrows = math.ceil(n_labels / ncols)
            fig, axes = plt.subplots(
                nrows,
                ncols,
                figsize=(3 * ncols, 3 * nrows),
                squeeze=False,
            )
            for ax in axes.flat:
                ax.axis("off")  # blank unused cells

            for plot_idx, (idx, label_name) in enumerate(selected):
                ax = axes[plot_idx // ncols, plot_idx % ncols]
                sample = dataset[idx]
                img = sample["img"].float().numpy()
                is_3d = img.ndim == 4
                img2d = img[0].mean(axis=projection_axis) if is_3d else img[0]
                ax.imshow(img2d, cmap="gray", vmin=-1, vmax=1)
                ax.set_title(label_name, fontsize=7)
                ax.axis("off")
                if "bbox" in sample:
                    try:
                        b = [float(v) for v in sample["bbox"]]
                        rect = _bbox_rect_for_projection(
                            b, img.shape[1:], projection_axis
                        )
                        if rect is not None:
                            ax.add_patch(
                                mpatches.Rectangle(
                                    *rect,
                                    linewidth=1,
                                    edgecolor="lime",
                                    facecolor="none",
                                )
                            )
                    except (TypeError, ValueError):
                        pass

            plt.tight_layout()
            yield fig
            return

        # ── normal mode: one figure per sample ────────────────────────────────
        for idx in selected:
            sample = dataset[idx]

            has_mask = "mask" in sample
            has_cls = "cls" in sample
            has_bbox = "bbox" in sample
            has_report = "report" in sample

            # col 0 : image (+ bbox overlay when present)
            # col 1 : mask overlay (when present)
            ncols = 1 + has_mask
            fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4), squeeze=False)
            ax_img = axes[0, 0]
            ax_mask = axes[0, 1] if has_mask else None

            # ── image ─────────────────────────────────────────────────────────
            img = sample["img"].float().numpy()  # (C, H, W) or (C, D, H, W)
            is_3d = img.ndim == 4
            img2d = img[0].mean(axis=projection_axis) if is_3d else img[0]  # → (H, W)

            ax_img.imshow(img2d, cmap="gray")  # , vmin=-1, vmax=1)
            ax_img.axis("off")

            # ── title from cls labels ─────────────────────────────────────────
            title_parts = [f"[{idx}]"]
            if is_3d:
                title_parts.append("(mean projection)")
            if has_cls:
                cls_vals = sample["cls"].float().tolist()
                pos = []
                for i, v in enumerate(cls_vals):
                    if v <= 0.5:
                        continue
                    grade = int(round(v))
                    pos.append(
                        f"{self.LABEL_COLS[i]} ({grade})" if grade > 1 else self.LABEL_COLS[i]
                    )
                title_parts.append(", ".join(pos) if pos else "no finding")
            ax_img.set_title("  ".join(title_parts), fontsize=8)

            # ── bbox overlay on image ─────────────────────────────────────────
            if has_bbox:
                try:
                    b = [float(v) for v in sample["bbox"]]
                    spatial = img.shape[1:]  # (H, W) or (D, H, W)
                    rect = _bbox_rect_for_projection(b, spatial, projection_axis)
                    if rect is not None:
                        ax_img.add_patch(
                            mpatches.Rectangle(
                                *rect, linewidth=1.5, edgecolor="lime", facecolor="none"
                            )
                        )
                        ax_img.set_title(ax_img.get_title() + "  [bbox]", fontsize=8)
                except (TypeError, ValueError):
                    pass

            # ── mask overlay ──────────────────────────────────────────────────
            if has_mask:
                mask = sample["mask"].float().numpy()  # (C, H, W) or (C, D, H, W)
                mask2d = (
                    mask[0].mean(axis=projection_axis) if mask.ndim == 4 else mask[0]
                )

                ax_mask.imshow(img2d, cmap="gray", vmin=-1, vmax=1)
                ax_mask.imshow(mask2d, cmap="Reds", alpha=0.5, vmin=0, vmax=1)
                ax_mask.set_title("mask overlay", fontsize=8)
                ax_mask.axis("off")

            # ── report path as figure footnote ────────────────────────────────
            if has_report:
                fig.text(
                    0.5,
                    0.01,
                    f"report: {sample['report']}",
                    ha="center",
                    fontsize=7,
                    color="gray",
                    wrap=True,
                )

            plt.tight_layout()
            yield fig

    def _get_harmonized_df(self):
        raise NotImplementedError

    def _infer_label_cols_from_df(self, df) -> list:
        return [c for c in df.columns if c not in _HARMONIZED_NON_LABEL_COLS]

    def _sync_label_cols_from_harmonizer(self, h) -> None:
        # Prefer actual column names from the harmonized df (always snake_case)
        # over the harmonizer's raw LABEL_COLS (which may still be title-case).
        df_cols = self._infer_label_cols_from_df(h.harmonized_df)
        if df_cols:
            self.LABEL_COLS = sorted(df_cols)
        elif not self.LABEL_COLS:
            raw = h.get_label_cols()
            self.LABEL_COLS = sorted(c.lower().replace(" ", "_") for c in raw)

    def _try_resolve_preset_harmonized(self):
        """Return a DataFrame if constructor presets apply, else ``None``."""
        if self._preset_harmonized_df is not None:
            df = self._preset_harmonized_df
            if not isinstance(df, pd.DataFrame):
                raise TypeError("harmonized_df must be a pandas DataFrame")
            df = df.copy()
            if not self.LABEL_COLS:
                self.LABEL_COLS = sorted(self._infer_label_cols_from_df(df))
            return df

        if self._harmonizer_save_path:
            hcls = self._HARMONIZER_CLS
            if hcls is None:
                raise ValueError(
                    f"{type(self).__name__} has no _HARMONIZER_CLS; "
                    "cannot use harmonizer_path=."
                )
            h = hcls.load_from_saved(self._harmonizer_save_path)
            self._sync_label_cols_from_harmonizer(h)
            return h.harmonized_df

        if self._user_harmonizer is not None:
            self._sync_label_cols_from_harmonizer(self._user_harmonizer)
            return self._user_harmonizer.harmonized_df.copy()

        return None

    def get_harmonized_df(self):
        """Return the harmonized metadata table (same as internal :meth:`_get_harmonized_df`)."""
        return self._get_harmonized_df()

    def set_harmonized_df(self, df) -> None:
        """Pin a harmonized DataFrame so subsequent splits use it without re-harmonizing."""
        self._preset_harmonized_df = df
        if df is not None and not self.LABEL_COLS:
            self.LABEL_COLS = sorted(self._infer_label_cols_from_df(df))

    def verify_images(self, drop_missing: bool = True) -> pd.DataFrame:
        """Check that every ``image_path`` exists on disk and optionally drop missing rows.

        Uses :attr:`base_image_dir` to resolve relative paths (same as
        :meth:`get_datasets`).

        Args:
            drop_missing: If ``True`` (default), rows with missing files are
                removed from the harmonized DataFrame.  If ``False``, the
                DataFrame is left unchanged and only the missing-files table
                is returned.

        Returns:
            DataFrame of rows whose image file was not found (empty if all
            files exist).
        """
        import os

        df = self._get_harmonized_df()

        def _exists(p):
            fp = os.path.join(self.base_image_dir, p) if self.base_image_dir else p
            return os.path.isfile(fp)

        from tqdm import tqdm

        tqdm.pandas(desc="Verifying images")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing images "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All image files exist.")
        else:
            print(f"Found {len(missing_df)} rows with missing images (not dropped).")

        return missing_df

    def get_datasets(
        self,
        n_splits: int = None,
        num_cores: int = 2,
        random_state: int = 56,
        train_transform=None,
        val_transform=None,
    ):
        """Harmonize and return a dataset, or a train/val split if n_splits is given.

        Args:
            n_splits: Number of folds; 1/n_splits of patients go to val.
                If ``None``, returns a single dataset over the full data.
            num_cores: Parallel workers for building data dicts.
            random_state: Seed for the patient-level train/val split (default: 56).
            train_transform: Optional MONAI transform for the training set.
                Overrides the dataset-level transform. Useful for augmentations
                that should only apply during training.
            val_transform: Optional MONAI transform for the validation set.
                Overrides the dataset-level transform.

        Returns:
            A single ``PersistentDataset`` when ``n_splits`` is ``None``, or a
            ``(train_dataset, val_dataset)`` tuple otherwise.
        """
        df = self._get_harmonized_df()
        self.LABEL_COLS = sorted(self.LABEL_COLS)
        cols = self._cols

        if n_splits is None:
            dicts = get_data_dict(
                df, self.base_image_dir, "image_path", cols, num_cores
            )
            return self._make_dataset(dicts, cache_subdir="")

        train_df, val_df = split_data(df, "patient_id", n_splits, random_state)

        train_dicts = get_data_dict(
            train_df, self.base_image_dir, "image_path", cols, num_cores
        )
        val_dicts = get_data_dict(
            val_df, self.base_image_dir, "image_path", cols, num_cores
        )

        return (
            self._make_dataset(train_dicts, "train", transform=train_transform),
            self._make_dataset(val_dicts, "val", transform=val_transform),
        )

    def get_folds(
        self,
        n_splits: int = 5,
        num_cores: int = 2,
        random_state: int = 56,
        train_transform=None,
        val_transform=None,
    ):
        """Yield (train_dataset, val_dataset) for each fold of a k-fold cross-validation.

        Patients are shuffled once and divided into ``n_splits`` equal buckets.
        Each iteration holds out one bucket as val and trains on the rest.

        Args:
            n_splits: Number of folds (default: 5).
            num_cores: Parallel workers for building data dicts.
            random_state: Seed for the patient shuffle (default: 56).
            train_transform: Optional MONAI transform override for training sets.
            val_transform: Optional MONAI transform override for validation sets.

        Yields:
            Tuple of (train_dataset, val_dataset) for each fold.
        """
        df = self._get_harmonized_df()
        self.LABEL_COLS = sorted(self.LABEL_COLS)
        cols = self._cols

        for fold, (train_df, val_df) in enumerate(
            kfold_splits(df, "patient_id", n_splits, random_state)
        ):
            train_dicts = get_data_dict(
                train_df, self.base_image_dir, "image_path", cols, num_cores
            )
            val_dicts = get_data_dict(
                val_df, self.base_image_dir, "image_path", cols, num_cores
            )

            yield (
                self._make_dataset(
                    train_dicts, f"fold_{fold}/train", transform=train_transform
                ),
                self._make_dataset(
                    val_dicts, f"fold_{fold}/val", transform=val_transform
                ),
            )
