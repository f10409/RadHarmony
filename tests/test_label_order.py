"""D-05: the label columns are sorted by their snake_case names. No data needed.

Each test writes a few fake source files with the real file names and the
real label columns, in the order of the source file. The expected order is
the D-05 rule (sort by snake_case name), not the order of the code.

Run with::

    .venv/bin/python -m pytest tests/test_label_order.py -v
"""
from __future__ import annotations

import pandas as pd
import pytest

import radharmony.dataset  # noqa: F401  (fills the registry)
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.harmonizer import (
    OpenICXRHarmonizer,
    VinDrCXRTestHarmonizer,
    VinDrCXRTrainHarmonizer,
    VinDrPCXRHarmonizer,
)
from radharmony.harmonizer.mrrate import PATHOLOGY_LABELS, MRRATEHarmonizer
from radharmony.registry import list_datasets, resolve_dataset


def _snake(name: str) -> str:
    return name.lower().replace(" ", "_")


def _expected(source_names) -> list[str]:
    """D-05: snake_case names, sorted."""
    return sorted(_snake(c) for c in source_names)


# The label columns of image_labels_train.csv (VinDr-CXR 1.0.0, PhysioNet), in file order.
VINDR_CXR_TRAIN = [
    "Aortic enlargement", "Atelectasis", "Calcification", "Cardiomegaly", "Clavicle fracture",
    "Consolidation", "Edema", "Emphysema", "Enlarged PA", "ILD", "Infiltration", "Lung Opacity",
    "Lung cavity", "Lung cyst", "Mediastinal shift", "Nodule/Mass", "Pleural effusion",
    "Pleural thickening", "Pneumothorax", "Pulmonary fibrosis", "Rib fracture", "Other lesion", "COPD",
    "Lung tumor", "Pneumonia", "Tuberculosis", "Other diseases", "No finding",
]


def _table_labels(h, df) -> list[str]:
    """The label columns of the table, in table order."""
    names = {_snake(c) for c in h.get_label_cols()}
    return [c for c in df.columns if c in names]


def _check(h, df, expected):
    assert _table_labels(h, df) == expected
    assert [_snake(c) for c in h.LABEL_COLS] == expected


def vindr_cxr_train(tmp_path):
    """2 images, 3 readers each. Returns the harmonizer."""
    rows = [{"image_id": i, "rad_id": f"R{r}", **{c: 1 for c in VINDR_CXR_TRAIN}}
            for i in ("a", "b") for r in (1, 2, 3)]
    pd.DataFrame(rows).to_csv(tmp_path / "image_labels_train.csv", index=False)
    return VinDrCXRTrainHarmonizer(csv_path=str(tmp_path / "image_labels_train.csv"))


def vindr_cxr_test(tmp_path):
    # The test file names one column "Other disease". The table calls it "other_diseases", as in train.
    cols = [c if c != "Other diseases" else "Other disease" for c in VINDR_CXR_TRAIN]
    pd.DataFrame([{"image_id": "a", **{c: 0 for c in cols}}]).to_csv(tmp_path / "test.csv", index=False)
    return VinDrCXRTestHarmonizer(csv_path=str(tmp_path / "test.csv"))


# The label columns of image_labels_train.csv (VinDr-PCXR 1.0.0), in file order.
VINDR_PCXR = [
    "No finding", "Bronchitis", "Brocho-pneumonia", "Other disease", "Bronchiolitis", "Situs inversus",
    "Pneumonia", "Pleuro-pneumonia", "Diagphramatic hernia", "Tuberculosis", "Congenital emphysema",
    "CPAM", "Hyaline membrane disease", "Mediastinal tumor", "Lung tumor",
]


def vindr_pcxr(tmp_path):
    for split in ("train", "test"):
        (tmp_path / split).mkdir()
        pd.DataFrame([{"image_id": f"{split}1", **{c: 0 for c in VINDR_PCXR}}]).to_csv(
            tmp_path / f"image_labels_{split}.csv", index=False)
    return VinDrPCXRHarmonizer(base_dir=str(tmp_path))


def openi(tmp_path):
    (tmp_path / "ecgen-radiology").mkdir()
    (tmp_path / "ecgen-radiology" / "1.xml").write_text(
        '<eCitation><uId id="CXR1"/><pmcId id="1"/>'
        '<MedlineCitation><Article><Abstract>'
        '<AbstractText Label="FINDINGS">Heart is large.</AbstractText>'
        '</Abstract></Article></MedlineCitation>'
        '<MeSH><major>Cardiomegaly</major></MeSH>'
        '<parentImage id="CXR1_1_IM-0001-3001"><figureId>F1</figureId></parentImage>'
        '</eCitation>')
    return OpenICXRHarmonizer(base_dir=str(tmp_path))


def mrrate(tmp_path):
    meta = tmp_path / "meta"
    (meta / "metadata").mkdir(parents=True)
    (meta / "pathology_labels").mkdir()
    pd.DataFrame({"patient_uid": ["p1"], "study_uid": ["s1"], "batch_id": ["b0"], "split": ["train"]}
                 ).to_csv(meta / "splits.csv", index=False)
    pd.DataFrame({"patient_uid": ["p1"], "study_uid": ["s1"], "series_id": ["t1w-raw-sag"],
                  "classified_modality": ["T1w"]}).to_csv(meta / "metadata" / "batch00_metadata.csv", index=False)
    pd.DataFrame([{"study_uid": "s1", **{c: 0 for c in PATHOLOGY_LABELS}}]).to_csv(
        meta / "pathology_labels" / "mrrate_labels.csv", index=False)
    return MRRATEHarmonizer(base_dir=str(tmp_path), with_masks=False)


def test_vindr_cxr_train(tmp_path):
    h = vindr_cxr_train(tmp_path)
    _check(h, h.harmonize(), _expected(VINDR_CXR_TRAIN))


def test_vindr_cxr_test(tmp_path):
    h = vindr_cxr_test(tmp_path)
    _check(h, h.harmonize(), _expected(VINDR_CXR_TRAIN))


def test_vindr_pcxr(tmp_path):
    h = vindr_pcxr(tmp_path)
    # The source names have "-": the table name is "brocho_pneumonia", not "brocho-pneumonia".
    _check(h, h.harmonize(), sorted(_snake(c).replace("-", "_") for c in VINDR_PCXR))


def test_openi(tmp_path):
    h = openi(tmp_path)
    df = h.harmonize()
    assert df["cardiomegaly"].tolist() == [1]
    # The 8 CheXpert labels that the harmonizer maps from MeSH (its docstring).
    _check(h, df, _expected(["No Finding", "Cardiomegaly", "Edema", "Atelectasis", "Consolidation",
                             "Pleural Effusion", "Pneumothorax", "Support Devices"]))


def test_mrrate(tmp_path):
    h = mrrate(tmp_path)
    _check(h, h.harmonize(), _expected(PATHOLOGY_LABELS))


# --- V-D-14: the LABEL_COLS of a dataset class name the cls vector -------------

def _datasets_with_labels():
    out = {}
    for key in sorted(list_datasets()):
        cls = resolve_dataset(key)
        if issubclass(cls, BaseRadiologicalDataset) and cls.LABEL_COLS:
            out.setdefault(cls.__name__, cls)
    return list(out.values())


@pytest.mark.parametrize("cls", _datasets_with_labels(), ids=lambda c: c.__name__)
def test_dataset_class_label_cols_are_in_cls_order(cls):
    """V-D-14: ``cls`` uses sorted snake_case names, so ``LABEL_COLS[j]`` must name ``cls[j]``."""
    assert cls.LABEL_COLS == _expected(cls.LABEL_COLS)
