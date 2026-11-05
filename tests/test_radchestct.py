"""RAD-ChestCT harmonizer on a small fake tree: the image folder argument (D-04)."""

import os

import pandas as pd
import pytest


def _tree(root):
    """2 scans in the valid split. Only trn1 has its .npz file."""
    (root / "images").mkdir()
    (root / "images" / "trn1.npz").write_bytes(b"")
    pd.DataFrame({"NoteAcc_DEID": ["trn1", "trn2"], "MRN_DEID": ["m1", "m2"]}).to_csv(
        root / "meta.csv", index=False)
    pd.DataFrame({"NoteAcc_DEID": ["trn1", "trn2"], "nodule*left_lung": ["1", "0"]}).to_csv(
        root / "imgvalid_Abnormality_and_Location_Labels.csv", index=False)
    return dict(csv_path=str(root / "meta.csv"),
                label_csv_path=str(root / "imgvalid_Abnormality_and_Location_Labels.csv"))


@pytest.mark.parametrize("arg", ["base_image_dir", "image_base_dir"])
def test_verify_images_uses_the_folder_of_the_harmonizer(tmp_path, arg):
    """V-H2-33 (D-04): verify_images() with defaults uses the image folder that the harmonizer got."""
    from radharmony.harmonizer import RadChestCTValidHarmonizer

    kw = _tree(tmp_path)
    folder = str(tmp_path / "images")
    h = RadChestCTValidHarmonizer(**kw, **{arg: folder})
    df = h.harmonize()
    assert df["image_path"].tolist() == ["trn1.npz"]  # trn2 has no file, so the harmonizer drops it
    assert h.base_image_dir == folder
    assert len(h.verify_images()) == 0


def test_save_and_load_keep_the_image_folder(tmp_path):
    """D-08: a saved harmonizer loads with the same image folder."""
    from radharmony.harmonizer import RadChestCTValidHarmonizer

    kw = _tree(tmp_path)
    h = RadChestCTValidHarmonizer(**kw, base_image_dir=str(tmp_path / "images"))
    h.harmonize()
    h.save(os.path.join(tmp_path, "h.pkl"))
    loaded = RadChestCTValidHarmonizer.load_from_saved(os.path.join(tmp_path, "h.pkl"))
    assert loaded.base_image_dir == loaded.image_base_dir == str(tmp_path / "images")


@pytest.mark.parametrize("new_name", ["base_image_dir", "image_base_dir"])
def test_an_old_saved_file_loads_with_a_new_folder(tmp_path, new_name):
    """D-08, V-H2-33: a file saved before 2026-11-03 holds image_base_dir. A new folder in either name wins."""
    import pickle

    from radharmony.harmonizer import RadChestCTValidHarmonizer

    kw = _tree(tmp_path)
    h = RadChestCTValidHarmonizer(**kw, base_image_dir=str(tmp_path / "images"))
    h.harmonize()
    path = tmp_path / "h.pkl"
    h.save(str(path))
    payload = pickle.loads(path.read_bytes())
    payload["init"]["image_base_dir"] = payload["init"].pop("base_image_dir")  # the old form
    path.write_bytes(pickle.dumps(payload))
    loaded = RadChestCTValidHarmonizer.load_from_saved(str(path), **{new_name: "/moved/images"})
    assert loaded.base_image_dir == loaded.image_base_dir == "/moved/images"


def test_two_different_folders_raise(tmp_path):
    """V-H2-33: base_image_dir and the old name image_base_dir must not name 2 folders."""
    from radharmony.harmonizer import RadChestCTValidHarmonizer

    kw = _tree(tmp_path)
    with pytest.raises(ValueError, match="image_base_dir"):
        RadChestCTValidHarmonizer(**kw, base_image_dir=str(tmp_path / "images"), image_base_dir=str(tmp_path))
