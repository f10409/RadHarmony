"""Harmonizer for the Emory CHORUS chest-radiograph subset.

Emory CHORUS is an internal Emory dataset combining an OMOP-CDM clinical
warehouse with a multi-modality DICOM tree.  This harmonizer targets the
curated **X-ray subset** (``images_dicom_xray/``) and, by default, filters it
down to **chest** radiographs only.

Unlike every other RadHarmony dataset, CHORUS ships **no metadata CSV** — the
study/series/image table has to be built by walking the DICOM tree and reading
each file's header.  That scan is expensive (tens of thousands of headers), so
the harmonizer caches the result to a manifest CSV (``csv_path``): the first
harmonize scans and writes it; later runs read it back.

Layout::

    <base_image_dir>/                 # e.g. .../Emory_CHORUS/images_dicom_xray
        <person_id>/Images/<StudyInstanceUID>/<SeriesInstanceUID>/<sop>.dcm

The harmonized frame carries:

- ``patient_id``  ← DICOM ``PatientID`` (== the top-level ``<person_id>`` folder)
- ``study_id``    ← ``StudyInstanceUID``
- ``series_id``   ← ``SeriesInstanceUID``
- ``image_path``  ← path to the ``.dcm`` **relative to** ``base_image_dir``
- ``view_position`` and metadata extras (``body_part``, ``study_date``, ``sex``,
  ``age``, ``manufacturer``) so downstream code can filter / stratify without a
  rescan.

No finding labels are produced (none ship with the subset; deriving them from
the OMOP condition codes is a separate, future step), and there are no masks,
bounding boxes, or reports.  The dataset is image-only.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from radharmony.harmonizer.base import BaseHarmonizer


#: Columns written to the manifest CSV (one row per image).
_MANIFEST_COLS = [
    "person_id",
    "study_uid",
    "series_uid",
    "sop_uid",
    "image_path",  # absolute at scan time, rewritten to relative in _scan_tree
    "modality",
    "view_position",
    "body_part",
    "study_description",
    "series_description",
    "study_date",
    "photometric",
    "rows",
    "columns",
    "sex",
    "age",
    "manufacturer",
]


def _read_chorus_header(args: tuple[str, str]) -> dict | None:
    """Read one DICOM header into a manifest row (module-level → picklable).

    Returns ``None`` if the file cannot be parsed, so unreadable files are
    skipped rather than aborting the whole scan.
    """
    person, path = args
    import pydicom

    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None

    def g(key: str) -> str:
        return str(getattr(ds, key, "") or "")

    return {
        "person_id": person,
        "study_uid": g("StudyInstanceUID"),
        "series_uid": g("SeriesInstanceUID"),
        "sop_uid": g("SOPInstanceUID"),
        "image_path": path,  # absolute; made relative to base_image_dir later
        "modality": g("Modality"),
        "view_position": g("ViewPosition"),
        "body_part": g("BodyPartExamined"),
        "study_description": g("StudyDescription"),
        "series_description": g("SeriesDescription"),
        "study_date": g("StudyDate"),
        "photometric": g("PhotometricInterpretation"),
        "rows": g("Rows"),
        "columns": g("Columns"),
        "sex": g("PatientSex"),
        "age": g("PatientAge"),
        "manufacturer": g("Manufacturer"),
    }


class EmoryCHORUSHarmonizer(BaseHarmonizer):
    """Harmonize the Emory CHORUS X-ray subset into the standard format.

    Args:
        base_image_dir: Root of the DICOM tree (the ``images_dicom_xray``
            directory containing the per-patient sub-folders).
        csv_path: Path to the manifest CSV used as a scan cache.  When the file
            exists it is read directly; otherwise the tree is scanned and the
            manifest written here (parent dirs created as needed).  When
            ``None``, the tree is scanned in memory on every ``harmonize()``
            (slow — prefer passing a path).
        chest_only: Keep only chest radiographs (``Modality`` in ``{DX, CR}`` and
            ``"CHEST"`` in the body-part / study-description).  The subset also
            contains abdomen, spine, skull, and extremity films; set ``False``
            to keep all of them.
        num_workers: Parallel worker processes for the header scan.
    """

    LABEL_COLS: list = []

    #: view_position arrives already named from the manifest; no rename source.
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None

    # series_id / view_position are core optional columns (populated below);
    # these are the extra metadata columns carried through for filtering.
    EXTRA_OUTPUT_COLS = [
        "body_part",
        "study_date",
        "sex",
        "age",
        "manufacturer",
    ]

    #: Modalities kept by the chest filter.
    _CHEST_MODALITIES = frozenset({"DX", "CR"})

    def __init__(
        self,
        base_image_dir: str,
        csv_path: str | None = None,
        chest_only: bool = True,
        num_workers: int = 12,
    ):
        super().__init__(csv_path=os.path.expanduser(csv_path) if csv_path else None)
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )
        self.chest_only = chest_only
        self.num_workers = num_workers

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        snap["chest_only"] = self.chest_only
        return snap

    # ------------------------------------------------------------------
    # Manifest scan / cache
    # ------------------------------------------------------------------

    def _scan_tree(self) -> pd.DataFrame:
        """Walk ``base_image_dir`` and read every DICOM header into a manifest."""
        if not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            raise ValueError(
                f"base_image_dir does not exist: {self.base_image_dir!r}"
            )

        tasks: list[tuple[str, str]] = []
        for entry in os.scandir(self.base_image_dir):
            if not entry.is_dir():
                continue
            person = entry.name
            for dirpath, _dirs, filenames in os.walk(entry.path):
                for fn in filenames:
                    if fn.lower().endswith(".dcm"):
                        tasks.append((person, os.path.join(dirpath, fn)))

        rows: list[dict] = []
        if self.num_workers and self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as ex:
                for r in ex.map(_read_chorus_header, tasks, chunksize=32):
                    if r is not None:
                        rows.append(r)
        else:
            for t in tasks:
                r = _read_chorus_header(t)
                if r is not None:
                    rows.append(r)

        df = pd.DataFrame(rows, columns=_MANIFEST_COLS)
        # Store paths relative to base_image_dir (the dataset re-joins the root).
        if not df.empty:
            df["image_path"] = df["image_path"].apply(
                lambda p: os.path.relpath(p, self.base_image_dir)
            )
        return df

    def _load_or_scan(self) -> pd.DataFrame:
        if self.csv_path and os.path.isfile(self.csv_path):
            return pd.read_csv(self.csv_path, dtype=str)
        df = self._scan_tree()
        if self.csv_path:
            parent = os.path.dirname(os.path.abspath(self.csv_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            df.to_csv(self.csv_path, index=False)
        return df

    def _filter_chest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep chest radiographs only (see ``chest_only``)."""
        modality = df["modality"].fillna("").str.upper()
        text = (
            df["body_part"].fillna("") + " " + df["study_description"].fillna("")
        ).str.upper()
        keep = modality.isin(self._CHEST_MODALITIES) & text.str.contains("CHEST")
        return df[keep].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["person_id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["study_uid"].astype(str)

    def _build_series_id(self) -> None:
        self.df["series_id"] = self.df["series_uid"].astype(str)

    def _build_image_path(self) -> None:
        # Already relative to base_image_dir from the scan / manifest.
        self.df["image_path"] = self.df["image_path"].astype(str)

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:  # noqa: ARG002
        self._harmonized_df_override = None
        self.df = self._load_or_scan()

        if self.chest_only:
            self.df = self._filter_chest(self.df)

        self._build_patient_id()
        self._build_study_id()
        self._build_series_id()
        self._build_image_path()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
