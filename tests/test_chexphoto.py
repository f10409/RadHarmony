"""CheXphoto harmonizer tests. Only the last test reads real data, and it skips when the data is not mounted.

CheXphoto looks close enough to CheXpert that inheriting its handling unchanged
"works" in the sense of raising nothing at all, while producing a corrupt frame.
Each test here pins one of those silent failures.

Run with::

    .venv/bin/python -m pytest tests/test_chexphoto.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

from radharmony.harmonizer import (
    CheXphotoHarmonizer,
    CheXphotoTrainHarmonizer,
    CheXphotoValidHarmonizer,
)

# The released header: underscored, and Path carries a CheXphoto-v1.0/<split>/
# prefix that the download tree does not have.
_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Enlarged_Cardiomediastinum", "Fracture", "Lung_Lesion", "Lung_Opacity",
    "No_Finding", "Pleural_Effusion", "Pleural_Other", "Pneumonia",
    "Pneumothorax", "Support_Devices",
]

_ROWS = [
    ("CheXphoto-v1.0/train/natural/iphone/patient00205/study1/view1_frontal.jpg", "AP"),
    ("CheXphoto-v1.0/train/natural/nokia/patient00205/study1/view1_frontal.jpg", "AP"),
    ("CheXphoto-v1.0/train/synthetic/digital/patient00623/study2/view1_frontal.jpg", "PA"),
    # AP_PA is legitimately null on lateral views — 96 such rows in real valid.csv.
    ("CheXphoto-v1.0/train/synthetic/photographic/patient00623/study2/view2_lateral.jpg", None),
]


@pytest.fixture()
def csv(tmp_path):
    return write_fake(tmp_path)


def write_fake(tmp_path):
    """4 images of 2 patients, with the released header. Returns the CSV path."""
    df = pd.DataFrame({
        "Path": [p for p, _ in _ROWS],
        "Sex": ["Female"] * 4,
        "Age": [55] * 4,
        "Frontal_Lateral": ["Frontal", "Frontal", "Frontal", "Lateral"],
        "AP_PA": [v for _, v in _ROWS],
    })
    for c in _LABELS:
        df[c] = 0.0
    df.loc[0, "Cardiomegaly"] = 1.0
    path = tmp_path / "train.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_patient_id_is_not_the_subset(csv):
    """CheXpert reads Path.split('/')[2]; here that segment is the subset.

    The inherited builder yields patient_id='natural'/'synthetic' for every row,
    collapsing the patient axis and silently breaking patient-level splitting.
    """
    df = CheXphotoHarmonizer(csv_path=csv).harmonize(drop_uncertain=False)
    assert set(df["patient_id"]) == {"patient00205", "patient00623"}
    assert not {"natural", "synthetic"} & set(df["patient_id"])


def test_study_id_is_patient_scoped(csv):
    df = CheXphotoHarmonizer(csv_path=csv).harmonize(drop_uncertain=False)
    assert set(df["study_id"]) == {"patient00205_study1", "patient00623_study2"}


def test_image_path_prefix_is_stripped(csv):
    """Without stripping, every path is wrong and no image ever loads."""
    df = CheXphotoHarmonizer(csv_path=csv).harmonize(drop_uncertain=False)
    assert not df["image_path"].str.contains("CheXphoto-v1.0").any()
    assert df["image_path"].iloc[0] == (
        "natural/iphone/patient00205/study1/view1_frontal.jpg"
    )


def test_underscored_label_columns_are_recognised(csv):
    df = CheXphotoHarmonizer(csv_path=csv).harmonize(drop_uncertain=False)
    for col in ("no_finding", "pleural_effusion", "support_devices",
                "enlarged_cardiomediastinum", "lung_opacity"):
        assert col in df.columns
    assert df["cardiomegaly"].sum() == 1.0


def test_subset_and_capture_method_are_derived(csv):
    df = CheXphotoHarmonizer(csv_path=csv).harmonize(drop_uncertain=False)
    assert list(df["subset"]) == ["natural", "natural", "synthetic", "synthetic"]
    assert list(df["capture_method"]) == [
        "iphone", "nokia", "digital", "photographic"
    ]


@pytest.mark.parametrize("subset,expected", [
    ("all", 4), ("natural", 2), ("synthetic", 2),
])
def test_subset_filter(csv, subset, expected):
    df = CheXphotoHarmonizer(csv_path=csv, subset=subset).harmonize(
        drop_uncertain=False)
    assert len(df) == expected


def test_drop_uncertain_ignores_non_label_nans(csv):
    """A null AP_PA is a lateral view, not an uncertain label.

    The CheXpert harmonizer drops on the label columns only, so the lateral row stays.
    """
    df = CheXphotoHarmonizer(csv_path=csv).harmonize(drop_uncertain=True)
    assert len(df) == 4


def test_uncertain_labels_still_drop(csv):
    frame = pd.read_csv(csv)
    frame.loc[0, "Pneumonia"] = -1.0
    frame.to_csv(csv, index=False)
    kept = CheXphotoHarmonizer(csv_path=csv).harmonize(drop_uncertain=True)
    assert len(kept) == 3


def test_split_subclasses_share_behaviour(csv):
    base = CheXphotoHarmonizer(csv_path=csv).harmonize(drop_uncertain=False)
    for cls in (CheXphotoTrainHarmonizer, CheXphotoValidHarmonizer):
        df = cls(csv_path=csv).harmonize(drop_uncertain=False)
        pd.testing.assert_frame_equal(base, df)


# --- V-H1-49 (D-09): a split class loads a file of the old class of its own split ---

def _old_file(tmp_path, split):
    """Save the old class on the fake ``<split>.csv``. Return the file path and its table."""
    src = tmp_path / f"{split}.csv"
    pd.read_csv(write_fake(tmp_path)).to_csv(src, index=False)
    old = CheXphotoHarmonizer(csv_path=str(src))
    table = old.harmonize(drop_uncertain=False)
    old.save(str(tmp_path / "old.pkl"))
    return str(tmp_path / "old.pkl"), table


def test_valid_class_loads_a_valid_file_of_the_old_class(tmp_path):
    path, table = _old_file(tmp_path, "valid")
    h = CheXphotoValidHarmonizer.load_from_saved(path)
    assert type(h) is CheXphotoValidHarmonizer
    pd.testing.assert_frame_equal(h.harmonized_df, table)
    with pytest.raises(ValueError, match="valid"):
        CheXphotoTrainHarmonizer.load_from_saved(path)


def test_train_class_loads_a_train_file_of_the_old_class(tmp_path):
    path, table = _old_file(tmp_path, "train")
    pd.testing.assert_frame_equal(CheXphotoTrainHarmonizer.load_from_saved(path).harmonized_df, table)
    with pytest.raises(ValueError, match="train"):
        CheXphotoValidHarmonizer.load_from_saved(path)


def test_old_file_of_another_csv_name_is_rejected(tmp_path):
    """The file name picks the split. A file of ``all.csv`` names no split, so the split class cannot check it."""
    path, _ = _old_file(tmp_path, "all")
    with pytest.raises(ValueError, match="no split"):
        CheXphotoValidHarmonizer.load_from_saved(path)


# --- V-H1-48: the counts in the docstring of the train class, from the raw CSV ---

def test_default_train_counts_on_the_real_csv():
    """The default drop_uncertain=True drops each row with -1 in a label: 20,034 images, 2,279 patients."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from dataset_paths import is_complete, paths

    root = paths("chexphoto_train", missing_ok=not is_complete()).get("base_image_dir", "")
    if not Path(root, "train.csv").is_file():
        pytest.skip("CheXphoto is not mounted")
    raw = pd.read_csv(Path(root, "train.csv"))
    keep = ~(raw[_LABELS] == -1).any(axis=1)
    patients = raw.loc[keep, "Path"].str.extract(r"(patient\d+)", expand=False)
    assert (int(keep.sum()), patients.nunique()) == (20_034, 2_279)
    df = CheXphotoTrainHarmonizer(csv_path=str(Path(root, "train.csv")), base_image_dir=root).harmonize()
    assert (len(df), df["patient_id"].nunique()) == (20_034, 2_279)
