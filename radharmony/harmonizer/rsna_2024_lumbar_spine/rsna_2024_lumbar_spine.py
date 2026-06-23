"""Harmonizer for the RSNA 2024 Lumbar Spine Degenerative Classification dataset.

Kaggle competition: ``rsna-2024-lumbar-spine-degenerative-classification``.

Source layout (after ``kagglehub.competition_download``)::

    <base_image_dir>/                        # train_images/
      <study_id>/
        <series_id>/
          1.dcm, 2.dcm, ...                  # one slice per file

    train.csv                                # 25 study-level severity columns
    train_series_descriptions.csv            # study_id, series_id, series_description
    train_label_coordinates.csv              # (study, series, instance, condition, level, x, y)

One row is emitted per ``(study_id, series_id)`` — 3 series per study on average
(Sagittal T1, Sagittal T2/STIR, Axial T2), and a study's labels are repeated
across its series.  The 25 string-valued targets (``Normal/Mild``, ``Moderate``,
``Severe``, NaN) are expanded into 75 binary columns of the form
``{condition}_{level}_{severity}`` so they plug natively into ``output_cls``.

Optionally, per-slice point annotations from ``train_label_coordinates.csv``
can be converted to tiny 3-D bboxes (cubes) centred on each annotated point,
so they show up as "dots" in the Gradio / bbox overlay pipeline.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from ..base import BaseHarmonizer


# The 25 study-level target columns as they appear in train.csv.
_BASE_TARGETS = [
    "spinal_canal_stenosis_l1_l2",
    "spinal_canal_stenosis_l2_l3",
    "spinal_canal_stenosis_l3_l4",
    "spinal_canal_stenosis_l4_l5",
    "spinal_canal_stenosis_l5_s1",
    "left_neural_foraminal_narrowing_l1_l2",
    "left_neural_foraminal_narrowing_l2_l3",
    "left_neural_foraminal_narrowing_l3_l4",
    "left_neural_foraminal_narrowing_l4_l5",
    "left_neural_foraminal_narrowing_l5_s1",
    "right_neural_foraminal_narrowing_l1_l2",
    "right_neural_foraminal_narrowing_l2_l3",
    "right_neural_foraminal_narrowing_l3_l4",
    "right_neural_foraminal_narrowing_l4_l5",
    "right_neural_foraminal_narrowing_l5_s1",
    "left_subarticular_stenosis_l1_l2",
    "left_subarticular_stenosis_l2_l3",
    "left_subarticular_stenosis_l3_l4",
    "left_subarticular_stenosis_l4_l5",
    "left_subarticular_stenosis_l5_s1",
    "right_subarticular_stenosis_l1_l2",
    "right_subarticular_stenosis_l2_l3",
    "right_subarticular_stenosis_l3_l4",
    "right_subarticular_stenosis_l4_l5",
    "right_subarticular_stenosis_l5_s1",
]

# Severity string -> suffix used for the one-hot column.
# Keep ``Normal/Mild`` collapsed to ``normal_mild`` (matches the Kaggle
# sample_submission ``normal_mild`` column).
_SEVERITY_SUFFIX = {
    "Normal/Mild": "normal_mild",
    "Moderate": "moderate",
    "Severe": "severe",
}


def _expanded_label_cols() -> list[str]:
    """Return the 75 snake_case one-hot label columns, sorted alphabetically."""
    cols = [
        f"{target}_{suffix}"
        for target in _BASE_TARGETS
        for suffix in _SEVERITY_SUFFIX.values()
    ]
    return sorted(cols)


class RSNA2024LumbarSpineHarmonizer(BaseHarmonizer):
    """Harmonize RSNA 2024 Lumbar Spine Degenerative Classification.

    Args:
        csv_path: Path to ``train.csv`` (25 study-level severity columns).
        series_description_csv_path: Path to ``train_series_descriptions.csv``
            (columns ``study_id``, ``series_id``, ``series_description``).
        coord_csv_path: Optional path to ``train_label_coordinates.csv``.
            When provided, each annotated point is converted to a tiny 3-D
            cube bbox and stored in the ``bbox`` column (with ``bbox_labels``
            = ``"<condition>_<level>"``). Requires ``base_image_dir`` to
            read image dimensions from DICOM headers.
        base_image_dir: Root of the DICOM tree — typically
            ``<kagglehub_cache>/competitions/rsna-2024-lumbar-spine-degenerative-classification/train_images/``.
        bbox_epsilon: Half-size of the in-plane (H, W) bbox (fraction of
            each axis).  Default 0.015 gives visible dots at 112³ volumes.
        bbox_z_epsilon: Half-size in the slice (D) direction.  ``None``
            (default) auto-sizes to ``0.5 / n_slices`` so adjacent-slice
            bboxes touch without overlapping.  Pass an explicit value to
            override (e.g. ``0.005`` for very thin slabs).
        num_workers: Parallel threads for DICOM header reads when building
            coordinate bboxes.  Default 8.
    """

    # 75 final one-hot columns (already snake_case).  BaseHarmonizer sorts
    # LABEL_COLS again during harmonize, so the order here is authoritative
    # only for reference.
    LABEL_COLS = _expanded_label_cols()

    LABEL_JOIN_COLS = None
    SERIES_ID_SOURCE_COL = "series_id"
    VIEW_POSITION_SOURCE_COL = "series_description"
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
        series_description_csv_path: str = None,
        coord_csv_path: str = None,
        base_image_dir: str = None,
        bbox_epsilon: float = 0.015,
        bbox_z_epsilon: float | None = None,
        num_workers: int = 8,
    ):
        super().__init__(csv_path=os.path.expanduser(csv_path) if csv_path else csv_path)
        self.series_description_csv_path = (
            os.path.expanduser(series_description_csv_path)
            if series_description_csv_path
            else series_description_csv_path
        )
        self.coord_csv_path = (
            os.path.expanduser(coord_csv_path) if coord_csv_path else coord_csv_path
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
        snap["series_description_csv_path"] = self.series_description_csv_path
        snap["coord_csv_path"] = self.coord_csv_path
        snap["base_image_dir"] = self.base_image_dir
        snap["bbox_epsilon"] = self.bbox_epsilon
        snap["bbox_z_epsilon"] = self.bbox_z_epsilon
        snap["num_workers"] = self.num_workers
        return snap

    # ------------------------------------------------------------------
    # Build hooks
    # ------------------------------------------------------------------

    def _build_patient_id(self) -> None:
        # No distinct patient id column; treat each study as its own patient so
        # patient-level splits keep all 3 series of a study in the same fold.
        self.df["patient_id"] = self.df["study_id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["study_id"].astype(str)

    def _build_image_path(self) -> None:
        # Relative to base_image_dir. Points at the series directory; the
        # ITKReader in RadiologyTransform3D loads the DICOM stack as a 3-D volume.
        self.df["image_path"] = (
            self.df["study_id"].astype(str) + "/" + self.df["series_id"].astype(str)
        )

    # ------------------------------------------------------------------
    # Coord CSV → tiny 3-D cube bboxes (one per annotated point)
    # ------------------------------------------------------------------

    def _build_bbox(self) -> None:
        """Convert ``train_label_coordinates.csv`` points to tiny cube bboxes.

        Each point has ``(study_id, series_id, instance_number, condition, level, x, y)``.
        We normalise to a unit cube whose axis 0 indexes the volume's slice
        direction in **MONAI dim 2 (= ITKReader's ascending-z) order**.  After
        :func:`_base_load_3d` runs ``_SITKOrientD`` + ``Transposed([0,3,2,1])``
        the bbox lands aligned with the reoriented image.

        * ``rank_in_ascending_z(instance_number) / n_instances`` → axis 0 (D, slice)
        * ``y / Rows``                                          → axis 1 (H)
        * ``x / Columns``                                       → axis 2 (W)

        Sorting by ``ImagePositionPatient[2]`` (rather than InstanceNumber)
        is critical because some series store DICOMs with InstanceNumber and
        z-position running in opposite directions; ITKReader always loads in
        ascending-z order, so anchoring rank on ``z`` keeps the harmonizer's
        bbox aligned with MONAI dim 2 voxel indices.

        Rows with no matching coords (or series without readable DICOMs)
        get empty lists — matches the convention used elsewhere.
        """
        if self.coord_csv_path is None:
            # No coord CSV → don't add bbox columns at all (backwards compatible
            # with the v0 harmonizer that didn't expose bboxes).
            return

        if not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            raise ValueError(
                "base_image_dir must point at an existing directory when "
                "coord_csv_path is set (DICOM headers are needed for "
                "bbox normalisation)."
            )

        coords = pd.read_csv(self.coord_csv_path)
        # Standardise for joining: string IDs (Kaggle ships them as ints).
        coords["study_id"] = coords["study_id"].astype(str)
        coords["series_id"] = coords["series_id"].astype(str)

        # Per-series probe: read every DICOM header, sort by z, build a
        # ``{InstanceNumber: ascending_z_rank}`` lookup so the harmonizer's
        # bbox z-coord matches MONAI dim 2 indexing regardless of how the
        # series stores its slice order.
        import pydicom

        def _probe_series(
            rel_path: str,
        ) -> tuple[str, tuple[dict[int, int], int, int, int] | None]:
            """Return (rel_path, (rank_of_inst, n_instances, Rows, Columns)) or None."""
            full = os.path.join(self.base_image_dir, rel_path)
            try:
                files = [f for f in os.listdir(full) if f.endswith(".dcm")]
                if not files:
                    return rel_path, None
                rows = cols = None
                inst_z: list[tuple[int, float]] = []
                for fname in files:
                    try:
                        ds = pydicom.dcmread(
                            os.path.join(full, fname),
                            stop_before_pixels=True,
                            specific_tags=[
                                "InstanceNumber",
                                "ImagePositionPatient",
                                "Rows",
                                "Columns",
                            ],
                        )
                        if rows is None:
                            rows, cols = int(ds.Rows), int(ds.Columns)
                        inst = int(ds.InstanceNumber)
                        z = float(ds.ImagePositionPatient[2])
                        inst_z.append((inst, z))
                    except Exception:
                        continue
                if not inst_z or rows is None:
                    return rel_path, None
                inst_z.sort(key=lambda t: t[1])  # ascending z = MONAI dim 2 order
                rank_of_inst = {inst: i for i, (inst, _) in enumerate(inst_z)}
                return rel_path, (rank_of_inst, len(inst_z), rows, cols)
            except Exception:
                return rel_path, None

        rel_paths = self.df["image_path"].tolist()
        probes: dict[str, tuple[dict[int, int], int, int, int]] = {}
        if self.num_workers > 1:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                for rel, dims in ex.map(_probe_series, rel_paths):
                    if dims is not None:
                        probes[rel] = dims
        else:
            for rel in rel_paths:
                _, dims = _probe_series(rel)
                if dims is not None:
                    probes[rel] = dims

        # Group coords by (study, series) for a fast per-row lookup.
        coord_groups: dict[tuple[str, str], pd.DataFrame] = dict(
            tuple(coords.groupby(["study_id", "series_id"]))
        )

        eps = self.bbox_epsilon
        eps_z_user = self.bbox_z_epsilon

        def _clip(v: float) -> float:
            return min(1.0, max(0.0, float(v)))

        def _row_bboxes(row) -> tuple[list, list]:
            # Cast both sides to str in case the merged df still has ints.
            key = (str(row["study_id"]), str(row["series_id"]))
            g = coord_groups.get(key)
            if g is None or g.empty:
                return [], []
            probe = probes.get(row["image_path"])
            if probe is None:
                return [], []
            rank_of_inst, n_inst, H, W = probe
            if n_inst <= 0 or H <= 0 or W <= 0:
                return [], []
            # Auto-size z half-thickness to half a slice when not specified, so
            # adjacent-slice bboxes touch but don't overlap.
            eps_z = eps_z_user if eps_z_user is not None else 0.5 / n_inst
            boxes, labels = [], []
            for _, p in g.iterrows():
                inst = int(p["instance_number"])
                rank = rank_of_inst.get(inst)
                if rank is None:
                    continue
                x_norm = p["x"] / W
                y_norm = p["y"] / H
                # Centre of the annotated slice in MONAI dim 2 order, [0, 1].
                z_norm = (rank + 0.5) / n_inst
                boxes.append([
                    _clip(z_norm - eps_z), _clip(z_norm + eps_z),
                    _clip(y_norm - eps), _clip(y_norm + eps),
                    _clip(x_norm - eps), _clip(x_norm + eps),
                ])
                labels.append(
                    (str(p["condition"]) + "_" + str(p["level"]))
                    .lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                )
            return boxes, labels

        bb_pairs = self.df.apply(_row_bboxes, axis=1)
        self.df["bbox"] = [b for b, _ in bb_pairs]
        self.df["bbox_labels"] = [l for _, l in bb_pairs]

    # ------------------------------------------------------------------
    # Custom harmonize — joins series into rows and expands labels
    # ------------------------------------------------------------------

    def harmonize(self, **kwargs) -> pd.DataFrame:
        """Read both CSVs, cross-join studies with their series, expand labels."""
        self._harmonized_df_override = None

        if self.series_description_csv_path is None:
            raise ValueError(
                "series_description_csv_path is required for "
                "RSNA2024LumbarSpineHarmonizer (train_series_descriptions.csv)."
            )

        labels_df = pd.read_csv(self.csv_path)
        series_df = pd.read_csv(self.series_description_csv_path)

        # One row per (study_id, series_id), carrying all 25 study-level label
        # columns and the series description.  inner join drops series whose
        # study is not in train.csv (shouldn't happen, but stays safe).
        df = series_df.merge(labels_df, on="study_id", how="inner")

        # Expand each string-valued target into 3 binary columns
        # (normal_mild / moderate / severe).  NaN rows become all-zero for that
        # target — ~1.2% of label cells across the dataset.
        for target in _BASE_TARGETS:
            vals = df[target]
            for severity, suffix in _SEVERITY_SUFFIX.items():
                df[f"{target}_{suffix}"] = (vals == severity).astype("int8")
        df = df.drop(columns=_BASE_TARGETS)

        self.df = df
        self._build_patient_id()
        self._build_study_id()
        # Cast series_id to str before _build_image_path consumes it.
        self._build_series_id()
        self._build_image_path()
        # Base class rename: series_description -> view_position.
        self._build_view_position()
        # No-op hooks for the unsupported optionals, included for symmetry.
        self._build_mask_path()
        self._build_bbox()
        self._build_report()

        self.LABEL_COLS = sorted(
            self.LABEL_COLS, key=lambda c: c.lower().replace(" ", "_")
        )
        self.df.dropna(subset=["image_path"], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

        return self._select_harmonized_columns(self.df)

    # ------------------------------------------------------------------
    # verify_images override — image_path points at a directory, not a file
    # ------------------------------------------------------------------

    def verify_images(
        self,
        base_image_dir: str = None,
        drop_missing: bool = True,
    ) -> pd.DataFrame:
        """Check series directories exist and contain at least one .dcm file.

        Overrides the base implementation because ``image_path`` here refers
        to a DICOM series directory, not a single file.
        """
        df = self.harmonized_df

        def _exists(p):
            fp = os.path.join(base_image_dir, p) if base_image_dir else p
            if not os.path.isdir(fp):
                return False
            # Cheap check: any .dcm file in the directory.
            try:
                for name in os.listdir(fp):
                    if name.endswith(".dcm"):
                        return True
            except OSError:
                return False
            return False

        from tqdm import tqdm
        tqdm.pandas(desc="Verifying series directories")
        mask = df["image_path"].progress_apply(_exists)
        missing_df = df[~mask].copy()

        if not missing_df.empty and drop_missing:
            kept = df[mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(missing_df)} rows with missing/empty series "
                f"({len(kept)} remaining)."
            )
        elif missing_df.empty:
            print("All series directories exist and contain DICOM files.")
        else:
            print(
                f"Found {len(missing_df)} rows with missing/empty series "
                "(not dropped)."
            )

        return missing_df
