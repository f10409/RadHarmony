"""Harmonizer for the RSNA 2022 Cervical Spine box subset.

A 3-D variant with one row per study. 235 of the 2,019 studies have at least one fracture
box. It reads ``train_bounding_boxes.csv`` and joins the 8 labels of ``train.csv``.

A box is ``[d_min, d_max, y_min, y_max, x_min, x_max]`` in the ``(D, H, W)`` order of
:func:`_base_load_3d`:

* ``D``: the slice position ``(rank + 0.5) / n_slices``. ``rank`` is the 0-based order of
  the slice among the ``.dcm`` files of the study, found at harmonize time.
* ``H``: ``y / Rows`` to ``(y + height) / Rows``.
* ``W``: ``x / Columns`` to ``(x + width) / Columns``.

``Rows`` and ``Columns`` come from the DICOM headers. Most studies are 512 x 512, one is
768 x 768. A study has many boxes. They go to ``bbox`` and ``bbox_labels`` as JSON text, so
a table saved with ``to_csv`` loads back (V-H3-34), as for NODE21.

:class:`RSNA2022CervicalSpineHarmonizer` has the same labels but no boxes.
"""

import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .._bbox import clip_boxes
from ..base import BaseHarmonizer
from .rsna_2022_cervical_spine import RSNA2022CervicalSpineHarmonizer


# Slice size when the study folder cannot be read (most studies are 512 x 512).
_SLICE_H = 512
_SLICE_W = 512


class RSNA2022CervicalSpineBboxHarmonizer(BaseHarmonizer):
    """Per-volume harmonizer for studies with fracture bounding boxes.

    One row per ``StudyInstanceUID``. ``image_path`` is the study folder, so ITKReader loads
    the full CT volume. Each slice box becomes a 3-D box around its slice.

    Args:
        csv_path: Path to ``train.csv`` (the 8 labels of each study).
        bbox_csv_path: Path to ``train_bounding_boxes.csv``
            (StudyInstanceUID, x, y, width, height, slice_number).
        base_image_dir: Root of the DICOM tree, usually ``train_images/``. It gives the slice
            count for the D coordinate. If ``None``, the count is ``max(slice_number) + 1``,
            which is less exact.
        bbox_epsilon: Not used. The in-plane (H, W) box comes from the source
            ``x``, ``y``, ``width`` and ``height``. A value gives a warning
            (V-H3-24). The argument stays, so a harmonizer saved before
            2026-11-03 still loads, with this warning.
        bbox_z_epsilon: Half-size in the slice (D) direction. ``None`` (default) gives
            ``0.5 / n_slices``, so boxes of next slices touch and do not overlap.
        num_workers: Number of threads for the folder scan.
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
        bbox_epsilon: float | None = None,
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
        if bbox_epsilon is not None:
            warnings.warn(
                "bbox_epsilon has no effect: the in-plane box comes from the source x, y, "
                "width and height (V-H3-24).",
                UserWarning,
                stacklevel=2,
            )
        self.bbox_z_epsilon = (
            None if bbox_z_epsilon is None else float(bbox_z_epsilon)
        )
        self.num_workers = int(num_workers)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_image_dir"] = self.base_image_dir
        snap["bbox_z_epsilon"] = self.bbox_z_epsilon
        snap["num_workers"] = self.num_workers
        return snap

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyInstanceUID"].astype(str)

    def _build_image_path(self) -> None:
        # A study folder, not one file.
        self.df["image_path"] = self.df["study_id"].astype(str)

    # image_path is a study folder, so the file check of the base drops every row (V-H3-22).
    verify_images = RSNA2022CervicalSpineHarmonizer.verify_images

    def _read_source(self) -> pd.DataFrame:
        """Return one row per study of the box CSV, with the 8 labels of ``train.csv``."""
        uids = pd.read_csv(self.bbox_csv_path)["StudyInstanceUID"].astype(str).unique()
        cls_df = pd.read_csv(self.csv_path)
        keep = ["StudyInstanceUID"] + self.LABEL_COLS
        return pd.DataFrame({"StudyInstanceUID": uids}).merge(cls_df[keep], on="StudyInstanceUID", how="left")

    def _probe_study(self, uid: str) -> tuple[str, list[int] | None, tuple[int, int] | None]:
        """Return ``(uid, slice_numbers_in_MONAI_dim2_order, (Rows, Columns))``.

        ITKReader orders the slices by ascending ``ImagePositionPatient`` z, so MONAI dim 2
        index 0 is the lowest z. This set stores the files in descending z order. A sort by
        file name gives the reverse order, and the box z values land at the wrong end of the
        volume. So the sort is by ``ImagePositionPatient[2]``, and the rank equals the MONAI
        dim 2 index. On 2026-10-28 this order matched the file order of the 3-D loader
        (``itk.GDCMSeriesFileNames``) for all 235 box studies. RSNA 2024 lumbar needs the loader
        order itself (V-H3-7).

        ``(Rows, Columns)`` is the slice size of the first header that has it, or ``None``.
        """
        if not self.base_image_dir:
            return uid, None, None
        study_dir = os.path.join(self.base_image_dir, uid)
        try:
            import pydicom

            stems_with_z: list[tuple[int, float]] = []
            size = None
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
                        specific_tags=["ImagePositionPatient", "Rows", "Columns"],
                    )
                    if size is None and "Rows" in ds and "Columns" in ds:
                        size = (int(ds.Rows), int(ds.Columns))
                    z = float(ds.ImagePositionPatient[2])
                except Exception:
                    z = float(sn)  # fall back to filename order
                stems_with_z.append((sn, z))
            if not stems_with_z:
                return uid, None, size
            stems_with_z.sort(key=lambda t: t[1])  # ascending z = MONAI dim 2 order
            return uid, [sn for sn, _ in stems_with_z], size
        except OSError:
            return uid, None, None

    def _build_bbox(self) -> None:
        """Build the 3-D boxes of each study in ``self.df``. Clip them (D-07) and store them as JSON text."""
        bbox_df = pd.read_csv(self.bbox_csv_path)
        bbox_df["StudyInstanceUID"] = bbox_df["StudyInstanceUID"].astype(str)

        # Only the studies left after the id filter.
        study_uids = self.df["StudyInstanceUID"].tolist()

        # Scan the study folders for the sorted slice numbers and the slice size.
        slice_index: dict[str, list[int] | None] = {}
        slice_size: dict[str, tuple[int, int] | None] = {}
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                probes = list(ex.map(self._probe_study, study_uids))
        else:
            probes = [self._probe_study(uid) for uid in study_uids]
        for uid, slices, size in probes:
            slice_index[uid] = slices
            slice_size[uid] = size

        eps_z_user = self.bbox_z_epsilon
        n_no_slice = 0  # boxes on a slice with no file in the study folder

        def _study_bboxes(uid: str, group: pd.DataFrame) -> list:
            nonlocal n_no_slice
            sorted_slices = slice_index.get(uid)
            if sorted_slices:
                n = len(sorted_slices)
                rank_of = {s: i for i, s in enumerate(sorted_slices)}
            else:
                # The folder is not readable.
                max_sn = int(group["slice_number"].max())
                n = max_sn + 1
                rank_of = {int(s): int(s) for s in group["slice_number"].unique()}

            eps_z = eps_z_user if eps_z_user is not None else 0.5 / n
            H, W = slice_size.get(uid) or (_SLICE_H, _SLICE_W)

            boxes = []
            for row in group.itertuples(index=False):
                sn = int(row.slice_number)
                rank = rank_of.get(sn)
                if rank is None:
                    n_no_slice += 1
                    continue
                d_center = (rank + 0.5) / n
                # clip_boxes clips the box to [0, 1] below (D-07).
                boxes.append([float(v) for v in (
                    d_center - eps_z, d_center + eps_z,  # D (slice)
                    row.y / H, (row.y + row.height) / H,  # H (rows)
                    row.x / W, (row.x + row.width) / W,   # W (cols)
                )])
            return boxes

        # One row per study. Each box marks a fracture, and the CSV has no class (V-H3-23).
        groups = dict(tuple(bbox_df.groupby("StudyInstanceUID")))
        boxes = [_study_bboxes(uid, groups[uid]) for uid in study_uids]
        if n_no_slice:
            warnings.warn(
                f"RSNA 2022 cervical spine: dropped {n_no_slice} box(es) on a slice with no DICOM file "
                "in the study folder (D-07).",
                UserWarning,
                stacklevel=2,
            )
        self.df["bbox"] = pd.Series(boxes, index=self.df.index, dtype=object)
        self.df["bbox_labels"] = pd.Series([["fracture"] * len(b) for b in boxes], index=self.df.index, dtype=object)

        clip_boxes(self.df, "RSNA 2022 cervical spine")  # D-07
        # JSON text, so that a CSV round trip keeps the boxes and labels (V-H3-34).
        for col in ("bbox", "bbox_labels"):
            # dtype=str, so that a table with no rows (an id filter) also has text columns, not float.
            self.df[col] = pd.Series([json.dumps(v) for v in self.df[col]], index=self.df.index, dtype=str)
