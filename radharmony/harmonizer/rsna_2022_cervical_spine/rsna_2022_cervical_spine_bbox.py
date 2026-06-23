"""Harmonizer for the RSNA 2022 Cervical Spine bounding-box subset.

Per-volume 3-D variant: one row per study (235 studies out of 2,019 have
at least one fracture-localizing bbox).  Reads ``train_bounding_boxes.csv``
and joins the per-study 8-binary-label table from ``train.csv``.

Bbox format: each box stored as
``[d_min, d_max, y_min, y_max, x_min, x_max]`` in the post-transpose
``(D, H, W)`` axis order used by :func:`_base_load_3d`:

* ``D`` — normalised slice position ``(rank + 0.5) / n_slices``.
  ``rank`` is the 0-based sort order of the annotated slice among all
  ``.dcm`` files in the study directory (probed at harmonization time).
* ``H`` — ``y / 512``  to  ``(y + height) / 512``  (row direction).
* ``W`` — ``x / 512``  to  ``(x + width) / 512``   (column direction).

Multiple bboxes per study are stored as a list of 6-element lists.

See also: :class:`RSNA2022CervicalSpineHarmonizer` for the 3-D per-volume
sibling that exposes the same 8 study-level labels but ignores bboxes.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..base import BaseHarmonizer


_SLICE_H = 512
_SLICE_W = 512


class RSNA2022CervicalSpineBboxHarmonizer(BaseHarmonizer):
    """Per-volume harmonizer for studies with fracture bounding boxes.

    One row per ``StudyInstanceUID`` — the image_path is the study directory
    so ITKReader loads the full CT volume.  Per-slice bbox annotations are
    converted to 3-D cubes centred on each annotated slice.

    Args:
        csv_path: Path to ``train.csv`` (provides the 8 cls labels per study).
        bbox_csv_path: Path to ``train_bounding_boxes.csv``
            (StudyInstanceUID, x, y, width, height, slice_number).
        base_image_dir: Root of the DICOM tree (typically ``train_images/``).
            Required for slice-count probing used in D-coordinate normalisation.
            When ``None``, falls back to ``max(slice_number) + 1`` as the
            denominator (less accurate but functional).
        bbox_epsilon: Half-size of the in-plane (H, W) bbox as a fraction
            of each axis.  Default 0.015 gives visible dots at 112³ volumes.
        bbox_z_epsilon: Half-size in the slice (D) direction.  ``None``
            (default) auto-sizes to ``0.5 / n_slices`` so adjacent-slice
            bboxes touch without overlapping.  Pass an explicit value to
            override.
        num_workers: Parallel threads for directory probing.
    """

    LABEL_COLS = [
        "C1",
        "C2",
        "C3",
        "C4",
        "C5",
        "C6",
        "C7",
        "patient_overall",
    ]
    LABEL_JOIN_COLS = ["StudyInstanceUID"]

    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    def __init__(
        self,
        csv_path: str,
        bbox_csv_path: str,
        base_image_dir: str = None,
        bbox_epsilon: float = 0.015,
        bbox_z_epsilon: float | None = None,
        num_workers: int = 8,
    ):
        super().__init__(
            csv_path=os.path.expanduser(csv_path) if csv_path else csv_path,
            bbox_csv_path=os.path.expanduser(bbox_csv_path) if bbox_csv_path else bbox_csv_path,
        )
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else base_image_dir
        )
        self.bbox_epsilon = float(bbox_epsilon)
        self.bbox_z_epsilon = (
            None if bbox_z_epsilon is None else float(bbox_z_epsilon)
        )
        self.num_workers = int(num_workers)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        snap["bbox_epsilon"] = self.bbox_epsilon
        snap["bbox_z_epsilon"] = self.bbox_z_epsilon
        snap["num_workers"] = self.num_workers
        return snap

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        # Directory path — ITKReader loads the whole DICOM stack as a volume.
        self.df["image_path"] = self.df["study_id"].astype(str)

    def _build_bbox(self) -> None:
        # No-op — bboxes are built directly in harmonize() where we have access
        # to the full per-slice data before grouping by study.
        pass

    # ------------------------------------------------------------------
    # Slice-count probing
    # ------------------------------------------------------------------

    def _probe_study(self, uid: str) -> tuple[str, list[int] | None]:
        """Return ``(uid, slice_numbers_in_MONAI_dim2_order)``.

        ITKReader loads slices ordered by ascending ImagePositionPatient z
        (so MONAI dim 2 index 0 = lowest z = most inferior).  When the
        DICOMs are *stored* in descending z order (file 1 = highest z, as
        on the RSNA 2022 cervical spine set), sorting by filename gives
        the **reverse** of MONAI dim 2 order and bbox z-coords land at the
        wrong end of the volume.  Sort by ImagePositionPatient[2] instead
        so harmonized rank == MONAI dim 2 voxel index.
        """
        if not self.base_image_dir:
            return uid, None
        study_dir = os.path.join(self.base_image_dir, uid)
        try:
            import pydicom

            stems_with_z: list[tuple[int, float]] = []
            for fname in os.listdir(study_dir):
                if not fname.endswith(".dcm"):
                    continue
                try:
                    sn = int(os.path.splitext(fname)[0])
                except ValueError:
                    continue
                try:
                    ds = pydicom.dcmread(
                        os.path.join(study_dir, fname),
                        stop_before_pixels=True,
                        specific_tags=["ImagePositionPatient"],
                    )
                    z = float(ds.ImagePositionPatient[2])
                except Exception:
                    z = float(sn)  # fall back to filename order
                stems_with_z.append((sn, z))
            if not stems_with_z:
                return uid, None
            stems_with_z.sort(key=lambda t: t[1])  # ascending z = MONAI dim 2 order
            return uid, [sn for sn, _ in stems_with_z]
        except OSError:
            return uid, None

    # ------------------------------------------------------------------
    # harmonize
    # ------------------------------------------------------------------

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read both CSVs, build 3-D bboxes per study, join cls labels."""
        self._harmonized_df_override = None

        bbox_df = pd.read_csv(self.bbox_csv_path)
        bbox_df["StudyInstanceUID"] = bbox_df["StudyInstanceUID"].astype(str)

        # Normalise 2-D pixel coords → [0, 1].
        bbox_df["y_min"] = (bbox_df["y"] / _SLICE_H).clip(0, 1)
        bbox_df["y_max"] = ((bbox_df["y"] + bbox_df["height"]) / _SLICE_H).clip(0, 1)
        bbox_df["x_min"] = (bbox_df["x"] / _SLICE_W).clip(0, 1)
        bbox_df["x_max"] = ((bbox_df["x"] + bbox_df["width"]) / _SLICE_W).clip(0, 1)

        study_uids = bbox_df["StudyInstanceUID"].unique().tolist()

        # Probe directories in parallel for sorted slice numbers.
        slice_index: dict[str, list[int] | None] = {}
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                for uid, slices in ex.map(self._probe_study, study_uids):
                    slice_index[uid] = slices
        else:
            for uid in study_uids:
                _, slices = self._probe_study(uid)
                slice_index[uid] = slices

        eps_z_user = self.bbox_z_epsilon

        def _clip(v: float) -> float:
            return min(1.0, max(0.0, float(v)))

        def _study_bboxes(uid: str, group: pd.DataFrame) -> list:
            sorted_slices = slice_index.get(uid)
            if sorted_slices:
                n = len(sorted_slices)
                rank_of = {s: i for i, s in enumerate(sorted_slices)}
            else:
                # Fallback when directory is unavailable.
                max_sn = int(group["slice_number"].max())
                n = max_sn + 1
                rank_of = {int(s): int(s) for s in group["slice_number"].unique()}

            # Auto-size z half-thickness to half a slice when not specified, so
            # adjacent-slice bboxes touch but don't overlap.
            eps_z = eps_z_user if eps_z_user is not None else 0.5 / n

            boxes = []
            for row in group.itertuples(index=False):
                sn = int(row.slice_number)
                rank = rank_of.get(sn)
                if rank is None:
                    continue
                d_center = (rank + 0.5) / n
                boxes.append([
                    _clip(d_center - eps_z), _clip(d_center + eps_z),  # D (slice)
                    float(row.y_min), float(row.y_max),                 # H (rows)
                    float(row.x_min), float(row.x_max),                 # W (cols)
                ])
            return boxes

        # One row per study.
        rows = [
            {"StudyInstanceUID": uid, "bbox": _study_bboxes(uid, grp)}
            for uid, grp in bbox_df.groupby("StudyInstanceUID", sort=False)
        ]
        per_study = pd.DataFrame(rows)

        self.df = per_study

        # Join 8 cls labels from train.csv.
        cls_df = pd.read_csv(self.csv_path)
        keep = ["StudyInstanceUID"] + self.LABEL_COLS
        self.df = self.df.merge(cls_df[keep], on="StudyInstanceUID", how="left")

        self._build_patient_id()
        self._build_study_id()
        self._build_image_path()
        self.LABEL_COLS = sorted(self.LABEL_COLS, key=lambda c: c.lower().replace(" ", "_"))
        self._build_labels()
        self._build_view_position()
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)
