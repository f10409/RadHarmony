"""Tests for ``radharmony.integrity.check_dataset`` on small synthetic datasets.

Each case puts one known problem in the samples. The expected status and
counts come from the samples that the test makes, not from the checker.

Run with::

    pytest tests/test_integrity.py -v
"""
from __future__ import annotations

import json

import pytest
import torch

from radharmony.integrity import check_dataset

N = 8


class _Fake:
    """Stand-in for a dataset class: ``get_datasets()`` returns the samples."""

    LABEL_COLS = ["a", "b"]

    def __init__(self, samples, **flags):
        self.samples = samples
        for k, v in flags.items():
            setattr(self, k, v)

    def get_datasets(self):
        return self.samples


def _img(seed: int = 0) -> torch.Tensor:
    return torch.rand(1, 8, 8, generator=torch.Generator().manual_seed(seed))


def _check(samples, **flags):
    return check_dataset(_Fake(samples, **flags), num_workers=0, batch_size=4)


def test_good_samples_are_ok():
    samples = [{"img": _img(i), "cls": torch.tensor([1.0, 0.0]),
                "bbox": [[0.1, 0.4, 0.2, 0.5]], "bbox_labels": ["nodule"]} for i in range(N)]
    r = _check(samples, output_cls=True, output_bbox=True)
    assert r.status == "OK", r.problems
    assert (r.success, r.bbox_count, r.bbox_invalid_count) == (N, N, 0)


# V-A-5: a bad value must change the status, not only a counter.
@pytest.mark.parametrize("name, samples, status", [
    ("nan image", [{"img": torch.full((1, 8, 8), float("nan"))} for _ in range(N)], "ERROR"),
    ("inf image", [{"img": torch.full((1, 8, 8), float("inf"))} for _ in range(N)], "ERROR"),
    ("box outside [0, 1]", [{"img": _img(), "bbox": [[1.5, 2.0, -0.3, 0.4]]} for _ in range(N)],
     "ERROR"),
    # Q-1 (D-39): the 2 examples of D-39, a wrong axis order and a box with no area.
    ("box min after max", [{"img": _img(), "bbox": [[0.6, 0.2, 0.4, 0.1]]} for _ in range(N)], "ERROR"),
    ("box with no area", [{"img": _img(), "bbox": [[0.5, 0.5, 0.3, 0.3]]} for _ in range(N)], "ERROR"),
    ("every label NaN", [{"img": _img(), "cls": torch.tensor([float("nan")] * 2)}
                         for _ in range(N)], "WARN"),
    ("every report empty", [{"img": _img(), "report": ""} for _ in range(N)], "WARN"),
    ("every mask empty", [{"img": _img(), "mask": torch.zeros(1, 8, 8)} for _ in range(N)],
     "WARN"),
])
def test_bad_values_change_the_status(name, samples, status):
    r = _check(samples)
    assert r.status == status, (name, r.problems)
    assert r.problems, name


def test_some_empty_masks_are_ok():
    """Negative images have an empty mask. That is normal data."""
    samples = [{"img": _img(i), "mask": torch.zeros(1, 8, 8) if i % 2 else torch.ones(1, 8, 8)}
               for i in range(N)]
    r = _check(samples, output_mask=True)
    assert r.status == "OK", r.problems
    assert r.mask_empty_count == N // 2


def test_requested_output_that_no_sample_has_warns():
    """``output_report=True``, but the samples have no ``report`` key (D-11)."""
    r = _check([{"img": _img()} for _ in range(N)], output_report=True)
    assert r.status == "WARN"
    assert any("report" in p for p in r.problems), r.problems


def test_boxes_with_no_labels_warn():
    """Q-2 (D-39): ``output_bbox=True``, and no sample has ``bbox_labels``."""
    r = _check([{"img": _img(), "bbox": [[0.1, 0.4, 0.2, 0.5]]} for _ in range(N)], output_bbox=True)
    assert r.status == "WARN"
    assert any("bbox_labels" in p for p in r.problems), r.problems


# V-A-8: a box list stored as a JSON string is parsed. An empty list has no box.
def test_json_string_boxes_are_parsed():
    samples = [{"img": _img(i),
                "bbox": json.dumps([[0.1, 0.4, 0.2, 0.5]] if i % 2 == 0 else [])}
               for i in range(6)]
    r = _check(samples)
    assert (r.bbox_count, r.bbox_invalid_count) == (3, 0)
    assert r.status == "OK", r.problems


def test_empty_box_list_has_no_box():
    r = _check([{"img": _img(), "bbox": []} for _ in range(N)])
    assert r.bbox_count == 0


# V-A-9: a later sample without a key does not crash the run.
def test_missing_key_in_a_later_sample_warns():
    samples = [{"img": _img(), "cls": torch.tensor([1.0, 0.0])}] + [{"img": _img()} for _ in range(3)]
    r = _check(samples)
    assert r.status == "WARN"
    assert r.success == 4
    assert r.missing_keys == {"cls": 3}
    assert any("'cls'" in p for p in r.problems), r.problems


# V-A-10: the range uses the finite pixels only, and the JSON report is strict JSON.
def test_range_ignores_nan_pixels():
    nan_img = torch.full((1, 8, 8), float("nan"))
    samples = [{"img": torch.zeros(1, 8, 8)}, {"img": nan_img}, {"img": torch.ones(1, 8, 8)}]
    r = _check(samples)
    assert (r.img_min, r.img_max) == (0.0, 1.0)
    assert r.img_nan_count == 1


def test_report_with_no_finite_pixel_is_strict_json():
    r = _check([{"img": torch.full((1, 8, 8), float("inf"))} for _ in range(4)])
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["img_min"] is None and d["img_max"] is None


# V-A-25: samples without ``img`` are failures, not silent skips.
def test_samples_without_img_are_an_error():
    samples = [{"image": torch.zeros(1, 8, 8), "cls": torch.tensor([1.0, 0.0])} for _ in range(N)]
    r = _check(samples)
    assert r.status == "ERROR"
    assert (r.success, r.failed) == (0, N)


@pytest.mark.parametrize("batch_size", [4, 8, 16])
def test_missing_key_count_does_not_depend_on_the_batches(batch_size):
    """V-A-9: samples 0-7 have no report, samples 8-15 have one."""
    samples = [{"img": _img(i)} for i in range(8)] + [{"img": _img(i), "report": "text"}
                                                      for i in range(8, 16)]
    r = check_dataset(_Fake(samples), num_workers=0, batch_size=batch_size)
    assert r.missing_keys == {"report": 8}
    assert r.status == "WARN"


@pytest.mark.parametrize("box", [None, float("nan"), "None", "nan", "null"])
def test_a_missing_box_value_is_no_box(box):
    """A row with no box annotation gives None or NaN, not a box."""
    samples = [{"img": _img(), "bbox": box}, {"img": _img(), "bbox": [[0.1, 0.4, 0.2, 0.5]]}]
    r = _check(samples)
    assert (r.bbox_count, r.bbox_invalid_count) == (1, 0)
    assert r.status == "OK", r.problems


def test_zero_samples_is_an_error():
    """Q-1 (D-39): a dataset with 0 samples is an ERROR, not a WARN."""
    report = _check([])
    assert report.status == "ERROR"
    assert "0 samples" in report.error_message
