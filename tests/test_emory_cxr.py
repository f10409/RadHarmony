"""Regression tests for the Emory CXR v2 harmonizers.

Synthetic fixtures only — these run without the dataset mounted. Each
test pins a failure mode that is silent in production: a join that
evaluates to nothing, a blank label read as a negative, a path that
stops being portable, or a DICOM row with no route back to a patient.
"""

import pandas as pd
import pytest

from radharmony.harmonizer import (
    EmoryCXRHarmonizer,
    EmoryCXRDeidHarmonizer,
    EmoryCXRDicomHarmonizer,
)


def _write_metadata(tmp_path, n=4):
    """Metadata CSV whose accession column pandas infers as int64."""
    rows = []
    for i in range(n):
        rows.append(
            {
                "empi_anon": 1000 + (i // 2),
                "AccessionNumber_anon": 900000000000000 + (i // 2),
                "StudyDate_anon": "2015-01-0%d" % (i + 1),
                "Sex": "F" if i % 2 else "M",
                "Age": 50 + i,
                "ImagePath": "%d/%d/sop%d.png" % (1000 + (i // 2), 900000000000000 + (i // 2), i),
                "Race": "White",
                "ViewPosition": "PA" if i % 2 == 0 else "Lateral",
                "InstitutionName_anon": "siteA",
                "Frontal": 1.0 if i % 2 == 0 else 0.0,
                "SOP": "sop%d" % i,
            }
        )
    p = tmp_path / "EmoryCXR_v2_Metadata_08152025.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def _write_reports(tmp_path, accessions):
    """Report CSV whose accession column pandas infers as *object*.

    The trailing multi-line quoted field is what defeats the fast path in
    the real file; a leading text row reproduces the same dtype here.
    """
    df = pd.DataFrame(
        {
            "AccessionNumber_anon": [str(a) for a in accessions],
            "deid_report": ["FINDINGS:\nClear lungs.\n\nIMPRESSION:\nNormal." for _ in accessions],
        }
    )
    p = tmp_path / "EmoryCXR_v2_Report_05102026.csv"
    df.to_csv(p, index=False)
    return p


def _write_labels(tmp_path, accessions):
    """Finding-label CSV with blanks and a -1, as the real table has."""
    df = pd.DataFrame(
        {
            "AccessionNumber_anon": list(accessions),
            "No Finding": [1.0] + [None] * (len(accessions) - 1),
            "Cardiomegaly": [None] + [1.0] * (len(accessions) - 1),
            "Pneumonia": [-1.0] + [None] * (len(accessions) - 1),
            "Enlarged Cardiomediastinum": [None] * len(accessions),
            "Lung Lesion": [None] * len(accessions),
            "Lung Opacity": [None] * len(accessions),
            "Edema": [None] * len(accessions),
            "Consolidation": [None] * len(accessions),
            "Atelectasis": [None] * len(accessions),
            "Pneumothorax": [None] * len(accessions),
            "Pleural Effusion": [None] * len(accessions),
            "Pleural Other": [None] * len(accessions),
            "Fracture": [None] * len(accessions),
            "Support Devices": [None] * len(accessions),
        }
    )
    p = tmp_path / "EmoryCXR_v2_FindingLabel_10162024.csv"
    df.to_csv(p, index=False)
    return p


def test_report_join_survives_the_accession_dtype_mismatch(tmp_path):
    """int64 metadata vs str reports must still join.

    Left un-normalised, pandas raises on the merge; the older failure
    mode (``isin`` against the raw columns) evaluates all-False and
    reports 0% coverage on a dataset whose true coverage is 100%.
    """
    meta = _write_metadata(tmp_path)
    accs = pd.read_csv(meta)["AccessionNumber_anon"].unique()
    reports = _write_reports(tmp_path, accs)

    df = EmoryCXRHarmonizer(
        csv_path=str(meta), base_image_dir="/data/DEID_PNG/", report_csv_path=str(reports)
    ).harmonize()

    assert len(df) == 4
    assert df["report"].notna().all(), "report join silently produced nothing"
    assert "IMPRESSION" in df["report"].iloc[0]


def test_blank_labels_stay_nan_and_uncertain_stays_minus_one(tmp_path):
    """A blank means 'unmentioned', not 'absent'.

    Filling NaN with 0 would invent negatives no radiologist wrote —
    ``No Finding`` has no explicit negatives at all in the real table.
    """
    meta = _write_metadata(tmp_path)
    accs = pd.read_csv(meta)["AccessionNumber_anon"].unique()
    df = EmoryCXRHarmonizer(
        csv_path=str(meta),
        base_image_dir="/data/DEID_PNG/",
        label_csv_path=str(_write_labels(tmp_path, accs)),
    ).harmonize()

    assert df["cardiomegaly"].isna().any(), "blank label was filled in"
    assert (df["pneumonia"] == -1).any(), "uncertain label was coerced away"
    assert (df["no_finding"] == 0).sum() == 0, "invented a negative"


def test_labels_left_join_keeps_unlabelled_studies(tmp_path):
    """An accession missing from the label table keeps its images.

    The base class's inner merge would drop them, which looks exactly
    like a smaller-than-expected cohort with nothing raising.
    """
    meta = _write_metadata(tmp_path)
    accs = pd.read_csv(meta)["AccessionNumber_anon"].unique()
    df = EmoryCXRHarmonizer(
        csv_path=str(meta),
        base_image_dir="/data/DEID_PNG/",
        label_csv_path=str(_write_labels(tmp_path, accs[:1])),  # only the first
    ).harmonize()

    assert len(df) == 4, "unlabelled studies were dropped by the label join"


def test_patient_id_is_coarser_than_study_id(tmp_path):
    """Splitting on study_id would leak: patients span studies."""
    meta = _write_metadata(tmp_path)
    df = EmoryCXRHarmonizer(
        csv_path=str(meta), base_image_dir="/data/DEID_PNG/"
    ).harmonize()
    assert df["patient_id"].nunique() < len(df)
    assert df["patient_id"].nunique() <= df["study_id"].nunique()


def test_view_position_follows_frontal(tmp_path):
    """Q-14 (D-01): Frontal wins over ViewPosition. Frontal 0 with AP and Frontal 1 with Lateral give NaN."""
    # The lateral name is LATERAL, the shared 2-D name (Q-13).
    pairs = [("PA", 1.0, "PA"), ("AP", 1.0, "AP"), ("Lateral", 0.0, "LATERAL"), ("PA", 0.0, "LATERAL"),
             ("AP", 0.0, None), ("Lateral", 1.0, None), (None, 0.0, "LATERAL"), (None, 1.0, None)]
    meta = _write_metadata(tmp_path, n=len(pairs))
    df = pd.read_csv(meta).assign(ViewPosition=[p[0] for p in pairs], Frontal=[p[1] for p in pairs])
    df.to_csv(meta, index=False)
    out = EmoryCXRHarmonizer(csv_path=str(meta), base_image_dir="/data/DEID_PNG/").harmonize()
    got = [None if pd.isna(v) else v for v in out.sort_values("image_path")["view_position"]]
    assert got == [p[2] for p in pairs]


def test_image_path_stays_relative(tmp_path):
    """A harmonized frame must survive being moved between mounts."""
    meta = _write_metadata(tmp_path)
    df = EmoryCXRHarmonizer(
        csv_path=str(meta), base_image_dir="/data/DEID_PNG/"
    ).harmonize()
    assert not df["image_path"].str.startswith("/").any()


def test_deid_harmonizer_refuses_the_original_png_tree(tmp_path):
    """ORIGINAL_PNG carries a different intensity normalisation."""
    meta = _write_metadata(tmp_path)
    with pytest.raises(ValueError, match="DEID_PNG"):
        EmoryCXRDeidHarmonizer(
            csv_path=str(meta), base_image_dir="/data/EmoryCXRv2/ORIGINAL_PNG/"
        )


# ----------------------------------------------------------------------
# DICOM variant
# ----------------------------------------------------------------------


def _write_mapping(tmp_path, sops):
    df = pd.DataFrame(
        {
            "dcm_path": [
                "/mnt/x/EmoryCXRv2/DEID_DICOM/B1/part01/%s/%s/%s.dcm" % (s, s, s)
                for s in sops
            ],
            "SOPInstanceUID": list(sops),
        }
    )
    p = tmp_path / "mapping_deid_dcm_sopid.csv"
    df.to_csv(p, index=False)
    return p


def test_dicom_variant_requires_the_mapping(tmp_path):
    """Without it a .dcm has no route back to a patient at all."""
    meta = _write_metadata(tmp_path)
    with pytest.raises(ValueError, match="mapping_csv_path is required"):
        EmoryCXRDicomHarmonizer(csv_path=str(meta), base_image_dir="/data/DEID_DICOM/")


def test_dicom_paths_are_rooted_at_the_dicom_tree(tmp_path):
    """The absolute prefix is stripped so the frame stays portable."""
    meta = _write_metadata(tmp_path)
    sops = pd.read_csv(meta)["SOP"].tolist()
    df = EmoryCXRDicomHarmonizer(
        csv_path=str(meta),
        base_image_dir="/data/DEID_DICOM/",
        mapping_csv_path=str(_write_mapping(tmp_path, sops)),
    ).harmonize()

    assert len(df) == 4
    assert not df["image_path"].str.startswith("/").any()
    assert df["image_path"].str.startswith("B1/").all()
    assert df["image_path"].str.endswith(".dcm").all()


def test_dicom_inner_join_drops_unmapped_sops(tmp_path):
    """The 3 undelivered files have no mapping row and no usable identity.

    They cannot be joined to any manifest, so dropping them is correct —
    keeping them would put a NaN path into the MONAI pipeline.
    """
    meta = _write_metadata(tmp_path)
    sops = pd.read_csv(meta)["SOP"].tolist()
    df = EmoryCXRDicomHarmonizer(
        csv_path=str(meta),
        base_image_dir="/data/DEID_DICOM/",
        mapping_csv_path=str(_write_mapping(tmp_path, sops[:2])),
    ).harmonize()

    assert len(df) == 2


def test_dicom_variant_rejects_a_malformed_mapping(tmp_path):
    meta = _write_metadata(tmp_path)
    bad = tmp_path / "mapping_deid_dcm_sopid.csv"
    pd.DataFrame({"dcm_path": ["/x/DEID_DICOM/a.dcm"]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="SOPInstanceUID"):
        EmoryCXRDicomHarmonizer(
            csv_path=str(meta),
            base_image_dir="/data/DEID_DICOM/",
            mapping_csv_path=str(bad),
        ).harmonize()
