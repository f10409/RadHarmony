"""Harmonizer for the MedSeg COVID-19 CT segmentation datasets.

MedSeg (medicalsegmentation.com) publishes them, and figshare mirrors them
(https://figshare.com/authors/MedSeg/9940190). They are the CT partner of the
CXR set QaTa-COV19. They are also the source of the "MosMedData+" benchmark in
the LViT and DD-CMD papers on text-guided segmentation. Use ``subset`` to
choose one of the two releases.

``subset="dataset1"``: MedSeg Covid Dataset 1 (figshare 13521488)::

    tr_im.nii.gz      (512, 512, 100)  100 annotated axial slices
    tr_mask.nii.gz    (512, 512, 100)  matching multi-class lesion masks
    val_im.nii.gz     (512, 512,  10)  10 more slices, no masks released
    Test-Images-Clinical-Details.csv   SIRM case, age and sex of each tr_im slice

  The CSV maps the 100 slices to 43 SIRM cases. ``patient_id`` is the SIRM case
  number (D-10). Only the first row of a case has the age and sex, and they
  apply to all slices of the case (D-06, D-26). Sex 0 is male and 1 is female.
  The row "Not used" lists SIRM cases, not slices, so the class skips it.

``subset="dataset2"``: MedSeg Covid Dataset 2 (figshare 13521509)::

    rp_im/{1..9}.nii.gz        9 whole volumes, for example (630, 630, 45)
    rp_msk/{1..9}.nii.gz       matching lesion masks
    rp_lung_msk/{1..9}.nii.gz  matching lung-field masks

  The volume index is a real patient group. ``patient_id`` is the volume
  number and ``study_id`` is ``<volume>_<slice>``. Split on ``patient_id``,
  because slices of one volume are strongly correlated.

Mask values are class indexes, not binary: 0 background, 1 ground-glass
opacity, 2 consolidation, 3 pleural effusion (dataset1 only, dataset2 stops at 2).

The 2-D pipeline cannot read a CT volume. So the harmonizer writes one PNG
for each slice, for the image and for the mask. The mask is a binary
any-lesion mask. The label columns say which classes are in each slice. Existing
PNGs are not rewritten. The PNGs go to
``<slice_output_dir>/<subset>_hu<low>_<high>/``. The window is in the folder
name, so PNGs of another window are never reused (V-H1-42). Each PNG has the
usual axial view, with rows along y and columns along x (D-12).

HU window: slices use a lung window (default ``[-1000, 400]`` HU) before the
8-bit step. This loses information on purpose. ``hu_window`` records it. The
raw Hounsfield values stay in the ``.nii.gz`` files.

Harmonized columns: ``patient_id``, ``study_id``, ``image_path`` and ``mask_path``
(absolute). dataset1 also has ``age_years`` and ``sex``. The labels are
``ground_glass``, ``consolidation``, ``pleural_effusion`` (NaN in dataset2, which
has no such class) and ``any_lesion``. The extra columns are ``subset``,
``volume``, ``slice_index`` and ``lesion_pixels``.
"""

import os

import numpy as np
import pandas as pd

from ..base import BaseHarmonizer

#: class index -> harmonized label column, per the MedSeg release notes
CLASS_TO_LABEL = {1: "ground_glass", 2: "consolidation", 3: "pleural_effusion"}

#: The classes that each subset annotates. A class that a subset does not annotate gives NaN (D-05).
SUBSET_CLASSES = {"dataset1": {1, 2, 3}, "dataset2": {1, 2}}

_LABEL_COLS = ["any_lesion", "consolidation", "ground_glass", "pleural_effusion"]

#: The dataset1 file that maps each tr_im slice to a SIRM case (D-10).
CLINICAL_CSV = "Test-Images-Clinical-Details.csv"
_CASE, _SLICE = "SIRM Case", "Slice position in .nii.gz"
_AGE, _SEX = "Patient age", "Patient Sex (Male = 0 Female = 1)"


def _read_slice_map(path: str) -> pd.DataFrame:
    """Read the dataset1 slice map.

    The age and sex are on the first row of each case only, so every slice of the case gets them.

    Returns:
        One row for each slice: ``slice_index``, ``sirm_case``, ``patient_age`` and ``patient_sex``.

    Raises:
        ValueError: If a source column is missing.
    """
    d = pd.read_csv(path)
    missing = {_CASE, _SLICE, _AGE, _SEX} - set(d.columns)
    if missing:
        raise ValueError(f"{path}: no column {sorted(missing)}")
    d = d[pd.to_numeric(d[_CASE], errors="coerce").notna()]  # blank rows and "Not used"
    case = d[_CASE].astype(float).astype(int).astype(str)
    first = d.groupby(case)[[_AGE, _SEX]].transform("first")
    return pd.DataFrame({
        "slice_index": d[_SLICE].astype(int),
        "sirm_case": case,
        "patient_age": first[_AGE],
        "patient_sex": first[_SEX].map({0: "M", 1: "F"}),
    })


def _load_volume(path: str) -> np.ndarray:
    """Read a NIfTI volume as (H, W, D) float32 without importing nibabel."""
    from monai.transforms import LoadImage

    return np.asarray(LoadImage(image_only=True)(path), dtype=np.float32)


def slice_folder(name: str, hu_window) -> str:
    """Return the folder name of the slice PNGs, for example ``dataset1_hu-1000_400`` (V-H1-42)."""
    lo, hi = hu_window
    return f"{name}_hu{lo:.10g}_{hi:.10g}"


def _axial(vol: np.ndarray, k: int) -> np.ndarray:
    """Return slice ``k`` in the usual axial view (rows along y). MONAI gives the array as (x, y, z) (D-12)."""
    return vol[:, :, k].T


def _window_to_uint8(sl: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip a Hounsfield slice to [lo, hi] and quantise to 8-bit."""
    x = np.clip(sl, lo, hi)
    x = (x - lo) / max(hi - lo, 1e-6)
    return (x * 255.0).round().astype(np.uint8)


class MedSegCOVIDCTHarmonizer(BaseHarmonizer):
    """Harmonize the MedSeg COVID-19 CT sets into the RadHarmony format.

    Args:
        base_dir: Folder of the release. For ``dataset1`` it holds
            ``tr_im.nii.gz`` and ``tr_mask.nii.gz``. For ``dataset2`` it holds
            ``rp_im/`` and ``rp_msk/``.
        slice_output_dir: Folder for the PNG slices. Required, because the
            2-D pipeline cannot read a CT volume.
        subset: ``"dataset1"`` or ``"dataset2"``.
        hu_window: ``(low, high)`` Hounsfield window before the 8-bit step.
            The default is ``(-1000, 400)`` (lung window).
    """

    LABEL_COLS = _LABEL_COLS

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["subset", "volume", "slice_index", "lesion_pixels"]
    #: From the dataset1 slice map. dataset2 has no age or sex, so the base steps skip it.
    AGE_SOURCE_COL = "patient_age"
    AGE_SOURCE_UNIT = "years"
    SEX_SOURCE_COL = "patient_sex"

    def __init__(self, base_dir: str, slice_output_dir: str = None,
                 subset: str = "dataset1", hu_window=(-1000.0, 400.0)):
        if subset not in ("dataset1", "dataset2"):
            raise ValueError(f"subset must be 'dataset1' or 'dataset2', got {subset!r}")
        self.base_dir = os.path.expanduser(base_dir)
        self.slice_output_dir = os.path.expanduser(slice_output_dir) if slice_output_dir else None
        self.subset = subset
        self.hu_window = tuple(hu_window)
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap.update({"base_dir": self.base_dir, "slice_output_dir": self.slice_output_dir,
                     "subset": self.subset, "hu_window": list(self.hu_window)})
        snap.pop("csv_path", None)  # not an argument of this class (V-H1-5)
        return snap

    # ------------------------------------------------------------------
    # Volume discovery
    # ------------------------------------------------------------------

    def _volume_pairs(self) -> list[tuple[str, str, str]]:
        """Return [(volume_name, image_path, mask_path)] for the chosen subset."""
        if self.subset == "dataset1":
            im = os.path.join(self.base_dir, "tr_im.nii.gz")
            mk = os.path.join(self.base_dir, "tr_mask.nii.gz")
            if not (os.path.isfile(im) and os.path.isfile(mk)):
                raise FileNotFoundError(
                    f"expected tr_im.nii.gz and tr_mask.nii.gz under {self.base_dir}. "
                    "For the val_im.nii.gz slices (no masks), use MedSegCOVIDCTValHarmonizer."
                )
            return [("tr", im, mk)]

        im_d, mk_d = os.path.join(self.base_dir, "rp_im"), os.path.join(self.base_dir, "rp_msk")
        if not (os.path.isdir(im_d) and os.path.isdir(mk_d)):
            raise FileNotFoundError(f"expected rp_im/ and rp_msk/ under {self.base_dir}")
        out = []
        for f in sorted(os.listdir(im_d), key=lambda s: int("".join(c for c in s if c.isdigit()) or 0)):
            if not f.endswith(".nii.gz"):
                continue
            m = os.path.join(mk_d, f)
            if os.path.isfile(m):
                out.append((f.split(".")[0], os.path.join(im_d, f), m))
        return out

    # ------------------------------------------------------------------
    # Slice materialisation
    # ------------------------------------------------------------------

    def _materialise(self) -> pd.DataFrame:
        from PIL import Image

        if not self.slice_output_dir:
            raise ValueError(
                "slice_output_dir is required. MedSeg ships CT volumes (.nii.gz); "
                "the 2-D pipeline needs one PNG per slice, written here once. "
                "Pass slice_output_dir='/path/to/medseg_slices'."
            )
        folder = os.path.join(self.slice_output_dir, slice_folder(self.subset, self.hu_window))
        img_d = os.path.join(folder, "images")
        msk_d = os.path.join(folder, "masks")
        os.makedirs(img_d, exist_ok=True)
        os.makedirs(msk_d, exist_ok=True)
        lo, hi = self.hu_window

        rows = []
        for vol_name, im_p, mk_p in self._volume_pairs():
            vol = _load_volume(im_p)
            msk = _load_volume(mk_p)
            if vol.shape != msk.shape:
                raise ValueError(f"{vol_name}: image {vol.shape} != mask {msk.shape}")
            n_slices = vol.shape[2]
            for k in range(n_slices):
                stem = f"{self.subset}_{vol_name}_{k:04d}"
                ip = os.path.join(img_d, stem + ".png")
                mp = os.path.join(msk_d, stem + ".png")
                m_sl = _axial(msk, k)
                if not os.path.isfile(ip):
                    Image.fromarray(_window_to_uint8(_axial(vol, k), lo, hi)).save(ip)
                if not os.path.isfile(mp):
                    # Binary any-lesion mask. The labels hold the class presence.
                    Image.fromarray(((m_sl > 0).astype(np.uint8) * 255)).save(mp)

                row = {"stem": stem, "image_path": os.path.basename(ip), "mask_path": mp,
                       "subset": self.subset, "volume": vol_name, "slice_index": k,
                       "lesion_pixels": int((m_sl > 0).sum()),
                       "any_lesion": int((m_sl > 0).any())}
                for cls, col in CLASS_TO_LABEL.items():
                    row[col] = int((m_sl == cls).any()) if cls in SUBSET_CLASSES[self.subset] else np.nan
                rows.append(row)
        df = pd.DataFrame(rows)
        return self._join_slice_map(df) if self.subset == "dataset1" else df

    def _join_slice_map(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the SIRM case, age and sex of each dataset1 slice from the slice map."""
        path = os.path.join(self.base_dir, CLINICAL_CSV)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"expected {CLINICAL_CSV} under {self.base_dir}. It maps each slice to its patient "
                "(figshare 13521488). Without it, one patient could be in 2 splits (D-10)."
            )
        df = df.merge(_read_slice_map(path), on="slice_index", how="left", validate="one_to_one")
        n = int(df["sirm_case"].isna().sum())
        if n:
            raise ValueError(f"{CLINICAL_CSV}: {n} tr_im slices have no SIRM case")
        return df

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _read_source(self) -> pd.DataFrame:
        """Write the slice PNGs and return one row per slice. There is no CSV."""
        return self._materialise()

    def preprocess_masks(self, *args, **kwargs) -> None:
        """Raise ``ValueError``. ``mask_path`` holds PNG masks already, not encoded masks to decode."""
        raise ValueError("MedSeg masks are PNG files already. Leave mask_output_dir as None.")

    def _build_patient_id(self) -> None:
        if self.subset == "dataset2":
            # the volume index is a genuine patient grouping here
            self.df["patient_id"] = self.df["volume"].astype(str)
        elif "sirm_case" in self.df.columns:
            self.df["patient_id"] = self.df["sirm_case"]
        else:
            # The val_im slices are not in the slice map, so each slice is its own patient.
            self.df["patient_id"] = self.df["stem"]

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["stem"]

    def _build_image_path(self) -> None:
        pass  # already set by _materialise (bare filename under images/)

    def _build_labels(self) -> None:
        # float 0/1 with NaN, as in the other harmonizers: an Int64 NA breaks the cls tensor (Q-12).
        for c in _LABEL_COLS:
            if c in self.df.columns:
                self.df[c] = self.df[c].astype("float64")
