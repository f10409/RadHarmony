"""Regression test for bbox axis alignment in the 3-D pipeline.

Origin story: in commit `d1d0535` (2026-04-22) the lumbar spine harmonizer
started emitting bbox coordinates in the post-Transpose ``(D, H, W)`` axis
order — i.e. the order the existing ``Transposed([0, 3, 2, 1])`` step in
``_base_load_3d`` produces.  In April 2026 a refactor that removed the
``Transposed`` (in service of adding ``OrientationD`` for cross-scanner
canonicalization) silently broke bbox alignment on lumbar spine — bboxes
ended up painted into the wrong axes and labels appeared on wrong vertebrae.

This test exists so the same regression can't slip past CI again.  It loads
one known-good lumbar spine sample with ``output_bbox=True`` through the
real pipeline and asserts that the painted ``bbox_mask`` lands in the slice
range we'd expect from the harmonizer's coord convention.

If this test starts failing, you are about to mis-align bboxes.  Read
``docs/proposals/orientation_normalization.md`` and commit ``d1d0535``
before "fixing" it.

Run with::

    .venv/bin/python -m pytest tests/test_bbox_axis_alignment.py -v
"""
from __future__ import annotations

import os
import warnings

import pytest
import torch

LUMBAR_BASE = (
    "/path/to/"
    "rsna-2024-lumbar-spine-degenerative-classification/train_images/"
)


@pytest.fixture(scope="module")
def lumbar_sample():
    """First lumbar-spine sample with bboxes enabled, fully pipelined."""
    if not os.path.isdir(LUMBAR_BASE):
        pytest.skip(f"lumbar spine data not present at {LUMBAR_BASE}")

    from radharmony.dataset import RSNA2024LumbarSpineDataset

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ds = RSNA2024LumbarSpineDataset(
            base_image_dir=LUMBAR_BASE,
            cache_dir=None,
            output_cls=True,
            output_bbox=True,
        )
        built = ds.get_datasets()
        return built[0]


def test_bbox_present(lumbar_sample):
    """Sample dict has bbox after the full pipeline."""
    assert "img" in lumbar_sample, "img key missing from sample"
    assert "bbox" in lumbar_sample, (
        "bbox missing from output — output_bbox plumbing or _MaskToBbox "
        "round-trip broken"
    )


def test_bbox_has_content(lumbar_sample):
    """At least a few bboxes survived the round-trip through bbox_mask."""
    bb = lumbar_sample["bbox"]
    n = bb.shape[0] if hasattr(bb, "shape") else len(bb)
    assert n >= 3, (
        f"only {n} bbox(es) survived the pipeline — "
        "lumbar spine should have many (level-specific spinal-canal-stenosis, "
        "neural foraminal narrowing, subarticular stenosis labels)"
    )


def test_bbox_axis_layout_matches_harmonizer_convention(lumbar_sample):
    """Lumbar spine bbox harmonizer (commit d1d0535) authors coords in
    post-Transpose (D, H, W) order: ``[z_min, z_max, y_min, y_max,
    x_min, x_max]`` per row.  For sagittal-series spinal-canal-stenosis
    points the harmonizer's verification reads:

      "5 spinal-canal-stenosis points cluster at z ~ 0.5 (middle slice),
       y climbs 0.34 -> 0.60 across L1/L2 -> L5/S1, x ~ 0.50 across all
       levels"

    If the load pipeline shuffles axes in a way that mis-paints bboxes
    (e.g. the Transposed([0, 3, 2, 1]) step gets removed, or an
    OrientationD slips in without compensating coord transformation),
    the z column drifts away from 0.5 and/or the y column stops being
    monotonic with vertebra level."""
    bb = lumbar_sample["bbox"]
    if not hasattr(bb, "shape"):
        pytest.skip("bbox is not a tensor — different format than expected")

    # Convert from bfloat16 / whatever to plain float for comparison
    bb = bb.float()

    # First 6-column slice = (z_min, z_max, y_min, y_max, x_min, x_max).
    # z (axis 0) midpoints — should cluster near 0.5
    z_mid = (bb[:, 0] + bb[:, 1]) / 2
    # x (axis 2) midpoints — should also cluster near 0.5 (midline sagittal)
    x_mid = (bb[:, 4] + bb[:, 5]) / 2

    # Generous tolerance: just want to catch "bboxes ended up in totally
    # wrong axis", not the exact voxel.
    z_med = z_mid.median().item()
    x_med = x_mid.median().item()

    assert 0.30 <= z_med <= 0.70, (
        f"bbox z-coord (axis 0) median is {z_med:.3f}, expected near 0.5.  "
        "If z drifted, the load pipeline has shuffled axes — likely the "
        "Transposed([0, 3, 2, 1]) step in _base_load_3d got removed, or an "
        "OrientationD/Spacing step is reorienting the bbox_mask differently "
        "from how _BboxToMask painted it.  See "
        "docs/PIPELINE_INVARIANTS.md and "
        "docs/proposals/orientation_normalization.md."
    )
    assert 0.30 <= x_med <= 0.70, (
        f"bbox x-coord (axis 2) median is {x_med:.3f}, expected near 0.5 "
        "for sagittal-series midline points.  Same axis-shuffling root cause "
        "as the z assertion above."
    )
