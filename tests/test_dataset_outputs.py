"""Tests for the samples of dataset classes and the rows that get_data_dict keeps (no NAS data).

Run with::

    pytest tests/test_dataset_outputs.py -v
"""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from radharmony.dataset import MontgomeryCXRDataset, RadiologyTransform2D, ShenzhenCXRDataset


def _gray_picture() -> np.ndarray:
    """A 48 x 64 gray picture: dark on the left, bright on the right, with some noise."""
    rng = np.random.default_rng(0)
    ramp = np.linspace(20, 230, 64)[None, :] * np.ones((48, 1))
    return np.clip(ramp + rng.normal(0, 5, ramp.shape), 0, 255).astype(np.uint8)


def test_shenzhen_image_has_one_channel_for_rgb_and_palette_png(tmp_path):
    """V-D-11: Shenzhen PNGs are palette (635) or RGB (27), all gray. Each sample is (1, 224, 224).

    The 3 files hold the same gray picture, so the 3 samples must be equal to the sample
    of the 1-channel file.
    """
    arr = _gray_picture()
    Image.fromarray(arr).save(tmp_path / "gray.png")
    Image.fromarray(np.stack([arr] * 3, axis=-1)).save(tmp_path / "rgb.png")
    palette = Image.fromarray(arr, mode="P")
    palette.putpalette([v for i in range(256) for v in (i, i, i)])
    palette.save(tmp_path / "palette.png")
    assert {Image.open(tmp_path / f).mode for f in ("gray.png", "rgb.png", "palette.png")} == {"L", "RGB", "P"}

    names = ["gray.png", "rgb.png", "palette.png"]
    df = pd.DataFrame({"patient_id": names, "study_id": names, "image_path": names, "tuberculosis": 0})
    ds = ShenzhenCXRDataset(
        harmonized_df=df, base_image_dir=str(tmp_path), cache_dir=None, dtype=torch.float32
    ).get_datasets(num_cores=1)
    imgs = [ds[i]["img"] for i in range(3)]
    assert [tuple(x.shape) for x in imgs] == [(1, 224, 224)] * 3
    torch.testing.assert_close(imgs[1], imgs[0])
    torch.testing.assert_close(imgs[2], imgs[0])


def test_augmentation_differs_across_workers_and_epochs(tmp_path):
    """V-D-7: with 2 DataLoader workers, the random flip and jitter differ per worker and per epoch.

    D-15: augmentation must be random each time. 8 rows hold the same image.
    """
    Image.fromarray(_gray_picture()).save(tmp_path / "img.png")
    df = pd.DataFrame({"patient_id": [f"p{i}" for i in range(8)], "image_path": ["img.png"] * 8, "tuberculosis": 0})
    t = (
        RadiologyTransform2D(img_size=32, output_keys={"img"}, dtype=torch.float32)
        .with_flip(spatial_axis=1, prob=0.5)
        .with_intensity_jitter(shift=0.3, scale=0.3, prob=1.0)
        .get_transform()
    )
    ds = MontgomeryCXRDataset(
        harmonized_df=df, base_image_dir=str(tmp_path), cache_dir=None, transform=t
    ).get_datasets(num_cores=1)

    def one_epoch():
        loader = DataLoader(ds, batch_size=1, num_workers=2, shuffle=False, timeout=120)
        return [round(float(b["img"].mean()), 4) for b in loader]

    epoch0, epoch1 = one_epoch(), one_epoch()
    # The 2 workers take turns: rows 0, 2, 4, 6 go to worker 0 and rows 1, 3, 5, 7 to worker 1.
    assert epoch0[0::2] != epoch0[1::2]
    assert epoch0 != epoch1


# Builds a dataset with a random flip and jitter. Prints 6 image sums. The first 3: seed 0 + ds[0],
# ds[0], seed 0 + ds[0]. The last 3: (seed 0 + DataLoader batch) x 2, seed 1 + ds[0].
_SEEDED_SAMPLES = """
import sys
import pandas as pd
import torch
from torch.utils.data import DataLoader
from radharmony.dataset import MontgomeryCXRDataset, RadiologyTransform2D

t = (RadiologyTransform2D(img_size=32, output_keys={"img"}, dtype=torch.float32)
     .with_flip(spatial_axis=1, prob=0.5).with_intensity_jitter(shift=0.3, scale=0.3, prob=1.0).get_transform())
df = pd.DataFrame({"patient_id": ["p0"], "image_path": ["img.png"], "tuberculosis": 0})
ds = MontgomeryCXRDataset(harmonized_df=df, base_image_dir=sys.argv[1], cache_dir=None, transform=t).get_datasets(num_cores=1)
out = []
torch.manual_seed(0)
out += [ds[0]["img"].sum(), ds[0]["img"].sum()]
torch.manual_seed(0)
out.append(ds[0]["img"].sum())
for _ in range(2):
    torch.manual_seed(0)
    out.append(next(iter(DataLoader(ds, batch_size=1, num_workers=0)))["img"].sum())
torch.manual_seed(1)
out.append(ds[0]["img"].sum())
print(" ".join(f"{float(v):.5f}" for v in out))
"""


def test_torch_manual_seed_repeats_the_augmentation_in_the_main_process(tmp_path):
    """Q-3 (D-15): `torch.manual_seed(0); ds[0]` gives the same augmentation in 2 new processes.

    The same holds for a DataLoader with num_workers=0, and for a second seed 0 with the same dataset.
    Without a new seed, the next ds[0] is a new augmentation. The jitter has prob 1, so 2 draws differ.
    The DataLoader takes 1 number from the PyTorch generator first, so its batch is not ds[0].
    """
    import subprocess
    import sys

    Image.fromarray(_gray_picture()).save(tmp_path / "img.png")
    runs = [
        subprocess.run([sys.executable, "-c", _SEEDED_SAMPLES, str(tmp_path)],
                       capture_output=True, text=True, check=True).stdout.split()
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    first, again, reseeded, loader, loader_again, seed1 = runs[0]
    assert first == reseeded
    assert loader == loader_again
    assert again != first
    assert seed1 != first


def test_a_sample_with_no_random_step_takes_no_number_from_the_torch_generator(tmp_path):
    """Q-3 (D-15): with no random step, ds[i] does not change the PyTorch random numbers of the user.

    Example: feature extraction with num_workers=0 and a seeded model gives the same numbers as before.
    """
    Image.fromarray(_gray_picture()).save(tmp_path / "img.png")
    df = pd.DataFrame({"patient_id": ["p0"], "image_path": ["img.png"], "tuberculosis": 0})
    t = RadiologyTransform2D(img_size=32, output_keys={"img"}, dtype=torch.float32).get_transform()
    ds = MontgomeryCXRDataset(
        harmonized_df=df, base_image_dir=str(tmp_path), cache_dir=None, transform=t
    ).get_datasets(num_cores=1)
    torch.manual_seed(0)
    expected = torch.rand(3)
    torch.manual_seed(0)
    ds[0]
    torch.testing.assert_close(torch.rand(3), expected)


def _table_with_gaps() -> pd.DataFrame:
    """5 images. Image 1 has no mask (NaN), image 2 an empty mask path, image 3 no report, image 4 a blank report."""
    return pd.DataFrame({
        "image_path": [f"img{i}.png" for i in range(5)],
        "mask_path": ["m0.png", np.nan, "", "m3.png", "m4.png"],
        "report": ["FINDINGS: clear.", "FINDINGS: clear.", "FINDINGS: clear.", np.nan, "  "],
    })


def test_requested_report_drops_rows_with_no_report_and_warns():
    """V-D-9 (D-33): with output_report=True, a row with no report (NaN or blank) is dropped, with a warning."""
    from radharmony.utils.data_utils import get_data_dict

    with pytest.warns(UserWarning, match="2 of 5 rows have no 'report'"):
        dicts = get_data_dict(_table_with_gaps(), "", "image_path", {"report": "report"}, num_cores=1)
    assert [d["img"] for d in dicts] == ["img0.png", "img1.png", "img2.png"]
    assert all(isinstance(d["report"], str) for d in dicts)


def test_requested_mask_drops_rows_with_no_mask_and_warns():
    """D-33: with output_mask=True, a row with an empty mask_path (NaN or "") is dropped, with a warning."""
    from radharmony.utils.data_utils import get_data_dict

    with pytest.warns(UserWarning, match="2 of 5 rows have no 'mask_path'"):
        dicts = get_data_dict(_table_with_gaps(), "", "image_path", {"mask": "mask_path"}, num_cores=2)
    assert [d["img"] for d in dicts] == ["img0.png", "img3.png", "img4.png"]


def test_output_not_requested_keeps_every_row():
    """D-33: a row is dropped only for an output that the user requests."""
    import warnings

    from radharmony.utils.data_utils import get_data_dict

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dicts = get_data_dict(_table_with_gaps(), "", "image_path", {}, num_cores=1)
    assert len(dicts) == 5


def test_blank_image_path_drops_the_row():
    """V-P-23: a row with a blank or NaN image_path has no file, so it is no sample."""
    from radharmony.utils.data_utils import get_data_dict

    df = pd.DataFrame({"image_path": ["a.png", "", np.nan, "  ", "c.png"]})
    dicts = get_data_dict(df, "/base/", "image_path", None, num_cores=1)
    assert [d["img"] for d in dicts] == ["/base/a.png", "/base/c.png"]
