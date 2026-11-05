"""Dataset integrity checking: shared by the verification notebook (human)
and ``scripts/check_datasets.py`` (agent / CI).

The notebook renders figures for a human to eyeball. The script emits a
machine-parseable JSON report and a non-zero exit code on failure.  Both drive
the *same* iteration + validation logic defined here, so the two never drift.

Everything in this module is dataset-agnostic. :func:`check_dataset` takes an
already-instantiated :class:`~radharmony.dataset.base.BaseRadiologicalDataset`.
It iterates every sample through a ``DataLoader``. It records image shape /
dtype / range and per-output-key statistics in an :class:`IntegrityReport`.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

__all__ = [
    "IntegrityReport",
    "check_dataset",
    "print_report",
    "subsample_dataset",
]


@dataclass
class IntegrityReport:
    """Statistics from iterating every sample in a dataset."""

    dataset: str = ""
    status: str = "OK"  # OK | WARN | ERROR
    error_message: str = ""
    total_time_s: float = 0.0

    # Counts
    total_samples: int = 0
    success: int = 0
    failed: int = 0
    failed_indices: list = field(default_factory=list)  # first N failed
    failed_errors: list = field(default_factory=list)  # corresponding error msgs
    failed_filenames: list = field(default_factory=list)  # corresponding file names

    # Image stats (from successful samples)
    img_shapes: dict = field(default_factory=dict)  # shape_str -> count
    img_dtype: str = ""
    img_min: float = float("inf")
    img_max: float = float("-inf")
    img_nan_count: int = 0
    img_inf_count: int = 0

    # Cls stats
    has_cls: bool = False
    cls_length: int = 0
    cls_all_nan_count: int = 0  # samples where every label is nan
    cls_pos_counts: dict = field(default_factory=dict)  # label_idx -> positive count

    # Mask stats
    has_mask: bool = False
    mask_shapes: dict = field(default_factory=dict)
    mask_empty_count: int = 0  # masks that are all zeros

    # Report stats
    has_report: bool = False
    report_count: int = 0
    report_empty_count: int = 0

    # BBox stats
    has_bbox: bool = False
    bbox_count: int = 0
    bbox_invalid_count: int = 0  # values outside [0, 1], wrong length, or a min not less than its max

    # Keys present
    sample_keys: list = field(default_factory=list)
    missing_keys: dict = field(default_factory=dict)  # key -> samples without it

    # Reasons for a WARN or ERROR status, one text each
    problems: list = field(default_factory=list)

    # Live reference to the dataset wrapper used in the integrity pass.  Kept
    # so the notebook can call ``ds_obj.visualize_samples(...)`` without
    # re-instantiating the wrapper.  Excluded from ``to_dict`` / JSON.
    ds_obj: object = None

    def summary_dict(self) -> dict:
        """Compact one-row summary for a stdout table."""
        shapes = ", ".join(f"{s}: {c}" for s, c in sorted(self.img_shapes.items()))
        return {
            "dataset": self.dataset,
            "status": self.status,
            "samples": self.total_samples,
            "success": self.success,
            "failed": self.failed,
            "img_range": (
                f"[{self.img_min:.3f}, {self.img_max:.3f}]" if self.has_range else "-"
            ),
            "img_nan": self.img_nan_count,
            "img_shapes": shapes,
            "time_s": f"{self.total_time_s:.1f}",
        }

    @property
    def has_range(self) -> bool:
        """True when at least one finite pixel was seen."""
        return math.isfinite(self.img_min)

    def to_dict(self, label_cols: list | None = None) -> dict:
        """Full JSON-serializable report.

        Args:
            label_cols: Optional label names. When given, ``cls_positive`` is
                keyed by name instead of column index.
        """
        d = asdict(self)
        d.pop("ds_obj", None)
        # inf sentinels are not valid JSON
        if not self.has_range:
            d["img_min"] = None
            d["img_max"] = None
        if label_cols:
            d["cls_positive"] = {
                (label_cols[i] if i < len(label_cols) else f"[{i}]"): c
                for i, c in sorted(self.cls_pos_counts.items())
            }
        return d


class _SafeDataset(Dataset):
    """Wraps a dataset so that per-sample errors are returned as
    ``(None, idx, err_str)`` instead of raising, letting DataLoader workers
    keep going through the rest of the samples."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        try:
            sample = self.dataset[i]
        except Exception as e:
            return None, i, f"{type(e).__name__}: {e}"
        if "img" not in sample:
            return None, i, "KeyError: the sample has no 'img' key"
        return sample, i, ""


def _collate(batch):
    """Collate a list of ``(sample_or_None, idx, err)`` tuples into a batch dict.

    A key that only some samples have holds the values of those samples.
    ``_present`` gives the number of samples with each key.
    """
    samples, indices, errors = zip(*batch, strict=True)
    out = {"_idx": list(indices), "_err": list(errors), "_present": {}}
    # only collate successful samples
    good = [s for s in samples if s is not None]
    if not good:
        return out

    keys = list(dict.fromkeys(k for s in good for k in s))
    for key in keys:
        vals = [s[key] for s in good if key in s]
        out["_present"][key] = len(vals)
        if isinstance(vals[0], torch.Tensor):
            try:
                out[key] = torch.stack(vals)
            except Exception:
                out[key] = vals  # variable shapes, fall back to list
        else:
            out[key] = vals
    return out


def subsample_dataset(ds, max_samples: int | None, random_state: int = 42):
    """Pin a small random subset of the harmonized table for a fast pass.

    Returns ``(n_before, n_after)`` so callers can decide whether to log.
    A no-op when ``max_samples`` is falsy or the table is already small.
    """
    df = ds.get_harmonized_df()
    n = len(df)
    if max_samples and n > max_samples:
        ds.set_harmonized_df(
            df.sample(n=max_samples, random_state=random_state).reset_index(drop=True)
        )
        return n, max_samples
    return n, n


def check_dataset(
    ds_obj,
    label_cols: list = None,
    max_errors: int = 20,
    num_workers: int = 8,
    batch_size: int = 8,
) -> IntegrityReport:
    """Instantiate a dataset via ``get_datasets()`` and iterate every sample.

    Args:
        ds_obj: An instantiated ``BaseRadiologicalDataset`` subclass.
        label_cols: Label column names for cls reporting. Defaults to
            ``ds_obj.LABEL_COLS``.
        max_errors: Max number of failed-sample details to keep.
        num_workers: DataLoader worker processes (0 = main process only).
        batch_size: Samples per batch.

    Returns:
        :class:`IntegrityReport` with per-sample statistics.  ``report.ds_obj``
        holds the dataset wrapper so plotting can call ``visualize_samples``
        without re-building it.
    """
    name = type(ds_obj).__name__
    report = IntegrityReport(dataset=name, ds_obj=ds_obj)
    label_cols = label_cols or getattr(ds_obj, "LABEL_COLS", [])

    try:
        t0 = time.time()
        dataset = ds_obj.get_datasets()
        build_time = time.time() - t0
    except Exception as e:
        report.status = "ERROR"
        report.error_message = f"get_datasets() failed: {e}"
        return report

    report.total_samples = len(dataset)
    if report.total_samples == 0:
        report.status = "ERROR"
        report.error_message = "Dataset has 0 samples."
        return report

    loader = DataLoader(
        _SafeDataset(dataset),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=_collate,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=False,
    )

    t0 = time.time()
    present: dict = {}  # key -> good samples with the key, over all batches

    for batch in tqdm(loader, desc=name):
        for idx, err in zip(batch["_idx"], batch["_err"], strict=True):
            if err:
                report.failed += 1
                if len(report.failed_indices) < max_errors:
                    report.failed_indices.append(int(idx))
                    report.failed_errors.append(err)
                    try:
                        fname = str(dataset.data[int(idx)]["img"])
                    except Exception:
                        fname = "N/A"
                    report.failed_filenames.append(fname)

        if "img" not in batch:
            continue  # entire mini-batch failed

        n_good = (
            batch["img"].shape[0]
            if isinstance(batch["img"], torch.Tensor)
            else len(batch["img"])
        )
        report.success += n_good

        # ── Keys: the union over all samples ──
        batch_keys = {k for k in batch if not k.startswith("_")}
        report.sample_keys = sorted(set(report.sample_keys) | batch_keys)
        for key, k in batch["_present"].items():
            present[key] = present.get(key, 0) + k
        report.has_cls = report.has_cls or "cls" in batch
        report.has_mask = report.has_mask or "mask" in batch
        report.has_report = report.has_report or "report" in batch
        report.has_bbox = report.has_bbox or "bbox" in batch
        if "cls" in batch and not report.cls_length:
            cls0 = batch["cls"]
            if isinstance(cls0, torch.Tensor):
                report.cls_length = cls0.shape[-1]
            elif hasattr(cls0[0], "__len__"):
                report.cls_length = len(cls0[0])
            else:
                report.cls_length = 1

        # ── img ──
        img = batch["img"]
        if isinstance(img, torch.Tensor):
            # img shape: (B, C, ...), per-sample shape is img.shape[1:]
            shape_str = str(tuple(img.shape[1:]))
            report.img_shapes[shape_str] = report.img_shapes.get(shape_str, 0) + n_good
            if not report.img_dtype:
                report.img_dtype = str(img.dtype)
            img_f = img.float()
            _update_range(report, img_f)
            if torch.isnan(img_f).any():
                report.img_nan_count += (
                    torch.isnan(img_f).flatten(1).any(1).sum().item()
                )
            if torch.isinf(img_f).any():
                report.img_inf_count += (
                    torch.isinf(img_f).flatten(1).any(1).sum().item()
                )
        else:  # list of tensors (variable shapes)
            for t in img:
                shape_str = str(tuple(t.shape))
                report.img_shapes[shape_str] = report.img_shapes.get(shape_str, 0) + 1
                if not report.img_dtype:
                    report.img_dtype = str(t.dtype)
                tf = t.float()
                _update_range(report, tf)
                if torch.isnan(tf).any():
                    report.img_nan_count += 1
                if torch.isinf(tf).any():
                    report.img_inf_count += 1

        # ── cls ──
        if "cls" in batch:
            cls_batch = batch["cls"]
            if isinstance(cls_batch, torch.Tensor):
                cls_f = cls_batch.float()  # (B, L)
                for row in cls_f:
                    vals = row.tolist()
                    if all(np.isnan(v) for v in vals):
                        report.cls_all_nan_count += 1
                    for idx, v in enumerate(vals):
                        if not np.isnan(v) and v > 0.5:
                            report.cls_pos_counts[idx] = (
                                report.cls_pos_counts.get(idx, 0) + 1
                            )
            else:
                for item in cls_batch:
                    vals = (
                        item.float().tolist()
                        if hasattr(item, "float")
                        else [float(v) for v in item]
                    )
                    if all(np.isnan(v) for v in vals):
                        report.cls_all_nan_count += 1
                    for idx, v in enumerate(vals):
                        if not np.isnan(v) and v > 0.5:
                            report.cls_pos_counts[idx] = (
                                report.cls_pos_counts.get(idx, 0) + 1
                            )

        # ── mask ──
        if "mask" in batch:
            masks = batch["mask"]
            items = (
                masks
                if isinstance(masks, list)
                else [masks[i] for i in range(masks.shape[0])]
            )
            for m in items:
                ms = str(tuple(m.shape))
                report.mask_shapes[ms] = report.mask_shapes.get(ms, 0) + 1
                if m.float().sum().item() == 0:
                    report.mask_empty_count += 1

        # ── report ──
        if "report" in batch:
            for rpt in batch["report"]:
                report.report_count += 1
                if not str(rpt).strip():
                    report.report_empty_count += 1

        # ── bbox ── per sample: a list/tensor of boxes. Each box is 4 (2-D)
        # or 6 (3-D) fractional coords. Descend to the per-box level.
        if "bbox" in batch:
            for sample_boxes in batch["bbox"]:
                boxes = (
                    sample_boxes.tolist()
                    if hasattr(sample_boxes, "tolist")
                    else sample_boxes
                )
                if isinstance(boxes, str):
                    # Some datasets keep the box list as a JSON string (NODE21).
                    try:
                        boxes = json.loads(boxes)
                    except ValueError:
                        # A row with no box can arrive as the text "None" or "nan".
                        boxes = None if boxes.strip() in ("None", "nan") else [boxes]
                if boxes is None or (isinstance(boxes, float) and math.isnan(boxes)):
                    continue  # this sample has no box
                if not isinstance(boxes, (list, tuple)):
                    boxes = [boxes]  # one value that is no box list: one invalid box
                if len(boxes) > 0 and not isinstance(boxes[0], (list, tuple)):
                    boxes = [boxes]  # a single flat box -> wrap
                for box in boxes:
                    report.bbox_count += 1
                    try:
                        b = [float(c) for c in (box.tolist() if hasattr(box, "tolist") else list(box))]
                        # Each min is less than its max (D-07). A wrong axis order breaks it (D-39, Q-1).
                        if len(b) not in (4, 6) or not all(0.0 <= c <= 1.0 for c in b) or not all(
                            b[i] < b[i + 1] for i in range(0, len(b), 2)
                        ):
                            report.bbox_invalid_count += 1
                    except Exception:
                        report.bbox_invalid_count += 1

    report.total_time_s = (time.time() - t0) + build_time
    report.missing_keys = {k: report.success - c for k, c in sorted(present.items())
                           if c < report.success}
    _set_status(report, ds_obj)
    return report


def _update_range(report: IntegrityReport, img: torch.Tensor) -> None:
    """Extend the pixel range with the finite pixels of ``img``."""
    finite = img[torch.isfinite(img)]
    if finite.numel():
        report.img_min = min(report.img_min, finite.min().item())
        report.img_max = max(report.img_max, finite.max().item())


def _set_status(report: IntegrityReport, ds_obj) -> None:
    """Set ``status`` and ``problems`` from the counts.

    ERROR, when a value breaks the contract:

    - no sample loads
    - an image has NaN or Inf pixels
    - a box is not 4 or 6 values in [0, 1], or a min is not less than its max

    WARN, when an output is not complete:

    - some samples do not load
    - some samples do not have a key
    - no sample has a requested output, or ``output_bbox=True`` and no sample has ``bbox_labels``
    - every label is NaN, every mask is empty, or every report is empty

    One empty mask or report is normal data. It does not change the status.
    """
    errors, warns = [], []
    n = report.success
    if report.failed:
        msg = f"{report.failed} of {report.total_samples} samples failed to load"
        (warns if n else errors).append(msg)
    if report.img_nan_count:
        errors.append(f"{report.img_nan_count} of {n} images have NaN pixels")
    if report.img_inf_count:
        errors.append(f"{report.img_inf_count} of {n} images have Inf pixels")
    if report.bbox_invalid_count:
        errors.append(f"{report.bbox_invalid_count} of {report.bbox_count} boxes "
                      "are not 4 or 6 values in [0, 1] with each min less than its max")
    for key, k in sorted(report.missing_keys.items()):
        warns.append(f"'{key}' is missing in {k} of {n} samples")
    for flag, key in (("output_cls", "cls"), ("output_mask", "mask"),
                      ("output_report", "report"), ("output_bbox", "bbox"),
                      ("output_bbox", "bbox_labels"), ("output_reg", "reg")):
        if n and bool(getattr(ds_obj, flag, False)) and key not in report.sample_keys:
            warns.append(f"{flag}=True, but no sample has '{key}'")
    n_cls = n - report.missing_keys.get("cls", 0)
    if report.has_cls and report.cls_all_nan_count == n_cls:
        warns.append("every label is NaN in every sample")
    n_mask = sum(report.mask_shapes.values())
    if report.has_mask and report.mask_empty_count == n_mask:
        warns.append("every mask is empty")
    if report.has_report and report.report_empty_count == report.report_count:
        warns.append("every report is empty")

    report.problems = errors + warns
    if errors:
        report.status = "ERROR"
    elif warns:
        report.status = "WARN"


def print_report(r: IntegrityReport, label_cols: list = None) -> None:
    """Pretty-print a single integrity report to stdout."""
    icon = {"OK": "✅", "WARN": "⚠️", "ERROR": "❌"}
    print(f"\n{'='*60}")
    print(f"{icon.get(r.status, '?')}  {r.dataset}  [{r.status}]")
    print(f"{'='*60}")

    if r.error_message:
        print(f"  Error: {r.error_message}")
    for p in r.problems:
        print(f"  Problem: {p}")
    if r.error_message and r.status == "ERROR" and r.success == 0:
        return

    print(f"  Total time     : {r.total_time_s:.1f}s")
    print(f"  Samples        : {r.total_samples:,}")
    print(f"  Success        : {r.success:,}")
    print(f"  Failed         : {r.failed:,}")
    print(f"  Sample keys    : {r.sample_keys}")
    print()

    print(f"  img dtype      : {r.img_dtype}")
    print(f"  img range      : [{r.img_min:.4f}, {r.img_max:.4f}]")
    print(f"  img NaN        : {r.img_nan_count:,} samples")
    print(f"  img Inf        : {r.img_inf_count:,} samples")
    print("  img shapes     :")
    for s, c in sorted(r.img_shapes.items(), key=lambda x: -x[1]):
        print(f"    {s:30s} x {c:,}")

    if r.has_cls:
        print(f"\n  cls vector len : {r.cls_length}")
        print(f"  cls all-NaN    : {r.cls_all_nan_count:,} samples")
        label_cols = label_cols or []
        print("  cls positive counts:")
        for idx in range(r.cls_length):
            name = label_cols[idx] if idx < len(label_cols) else f"[{idx}]"
            pos = r.cls_pos_counts.get(idx, 0)
            pct = pos / r.success * 100 if r.success else 0
            print(f"    {name:40s}  {pos:>6,} / {r.success:,} ({pct:5.1f}%)")

    if r.has_mask:
        print("\n  mask shapes    :")
        for s, c in sorted(r.mask_shapes.items(), key=lambda x: -x[1]):
            print(f"    {s:30s} x {c:,}")
        print(f"  mask all-zero  : {r.mask_empty_count:,} / {r.success:,}")

    if r.has_report:
        print(f"\n  reports        : {r.report_count:,}")
        print(f"  reports empty  : {r.report_empty_count:,}")

    if r.has_bbox:
        print(f"\n  bboxes         : {r.bbox_count:,}")
        print(f"  bboxes invalid : {r.bbox_invalid_count:,}")

    if r.failed_indices:
        print(f"\n  First {len(r.failed_indices)} failures:")
        for idx, err, fname in zip(
            r.failed_indices, r.failed_errors, r.failed_filenames,
            strict=True,
        ):
            print(f"    [{idx}] {fname}")
            print(f"           {err}")
