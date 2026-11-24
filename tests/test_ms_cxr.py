"""MS-CXR harmonizer on a small fake CSV (MS-CXR is restricted): box values (D-07)."""

import io
import json

import pandas as pd


def _csv(path):
    """1 image with 1 box. The values are the first real MS-CXR box: x=196, y=1136, w=532, h=315."""
    pd.DataFrame({
        "dicom_id": ["d1"], "path": ["files/p10/p10233088/s54276838/d1.jpg"],
        "category_name": ["Pneumonia"], "label_text": ["right lower lobe pneumonia"],
        "x": [196], "y": [1136], "w": [532], "h": [315],
        "image_width": [1767], "image_height": [2021], "split": ["test"],
    }).to_csv(path, index=False)


def test_box_values_are_python_floats_and_survive_a_csv_round_trip(tmp_path):
    """V-H2-35: a box holds Python floats, so a CSV export keeps a JSON box that json.loads reads."""
    from radharmony.harmonizer import MSCXRHarmonizer

    _csv(tmp_path / "ms_cxr.csv")
    df = MSCXRHarmonizer(csv_path=str(tmp_path / "ms_cxr.csv")).harmonize()
    box = df["bbox"].iloc[0]
    assert all(type(v) is float for b in box for v in b)
    # [y_min, y_max, x_min, x_max] as fractions of the image (D-07).
    assert box == [[1136 / 2021, (1136 + 315) / 2021, 196 / 1767, (196 + 532) / 1767]]
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    assert json.loads(pd.read_csv(buf)["bbox"].iloc[0]) == box
