"""Harmonizer for the RSNA 2024 Lumbar Spine Degenerative Classification dataset.

Kaggle competition ``rsna-2024-lumbar-spine-degenerative-classification``. Files:
``train_images/<study_id>/<series_id>/<n>.dcm`` (one slice per file), ``train.csv`` (25 study
severity columns), ``train_series_descriptions.csv`` and ``train_label_coordinates.csv``.

The table has one row for each ``(study_id, series_id)``, about 3 series for each study
(Sagittal T1, Sagittal T2/STIR, Axial T2). A study's labels repeat on its series. The 25
targets (``Normal/Mild``, ``Moderate``, ``Severe`` or NaN) become 75 binary columns
``{condition}_{level}_{severity}`` for ``output_cls``.

With ``train_label_coordinates.csv``, each annotated point becomes a tiny 3-D box (a cube
around the point). It shows as a dot in the Gradio box overlay.

Details: ``docs/wiki/reference/datasets/ct-mri-other/rsna_2024_lumbar_spine.md``.
"""

import os
import warnings
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .._bbox import clip_boxes
from ..base import BaseHarmonizer


# The 25 target columns of train.csv.
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

# Severity text and the suffix of its 0/1 column. ``normal_mild`` is the name in the Kaggle
# sample submission.
_SEVERITY_SUFFIX = {
    "Normal/Mild": "normal_mild",
    "Moderate": "moderate",
    "Severe": "severe",
}


def _expanded_label_cols() -> list[str]:
    """Return the 75 snake_case label columns, sorted."""
    cols = [
        f"{target}_{suffix}"
        for target in _BASE_TARGETS
        for suffix in _SEVERITY_SUFFIX.values()
    ]
    return sorted(cols)


def _loader_file_order(series_dir: str, series_file_names) -> list[str]:
    """Return the DICOM files of a series in the slice order of the 3-D loader.

    ``ITKReader`` (MONAI) lists the files with ``itk.GDCMSeriesFileNames`` and stacks them in
    that order. This is not always ascending ``ImagePositionPatient[2]``. About half of the
    series stack the slices the other way (V-H3-7). So the box rank uses the same call as the
    loader.

    Args:
        series_dir: the series folder.
        series_file_names: the class ``itk.GDCMSeriesFileNames``. The caller gets it in the
            main thread, because ``itk`` loads its classes lazily and that fails in a worker
            thread.

    Returns:
        The file paths, in the order of the slices in the loaded volume.
    """
    names = series_file_names.New()
    names.SetUseSeriesDetails(True)
    names.AddSeriesRestriction("0008|0021")  # Series Date, as in ITKReader
    names.SetDirectory(series_dir)
    uids = names.GetSeriesUIDs()
    return list(names.GetFileNames(uids[0])) if uids else []


class RSNA2024LumbarSpineHarmonizer(BaseHarmonizer):
    """Harmonize RSNA 2024 Lumbar Spine Degenerative Classification.

    Args:
        csv_path: Path to ``train.csv`` (25 study severity columns).
        series_description_csv_path: Path to ``train_series_descriptions.csv``
            (columns ``study_id``, ``series_id``, ``series_description``).
        coord_csv_path: Path to ``train_label_coordinates.csv``. Optional. Each point becomes
            a tiny 3-D box in ``bbox``, with ``bbox_labels`` = ``"<condition>_<level>"``.
            It needs ``base_image_dir``, for the image size in the DICOM headers.
        base_image_dir: Root of the DICOM tree, usually
            ``<kagglehub_cache>/competitions/rsna-2024-lumbar-spine-degenerative-classification/train_images/``.
        bbox_epsilon: Half-size of the in-plane (H, W) box, as a fraction of each axis.
            The default 0.015 gives visible dots at 112^3 volumes.
        bbox_z_epsilon: Half-size in the slice (D) direction. ``None`` (default) gives
            ``0.5 / n_slices``, so boxes of next slices touch and do not overlap. For example,
            use ``0.005`` for very thin boxes.
        num_workers: Number of threads for the DICOM header reads of the box step.
    """

    #: view_position is a 3-D plane or series name, not a 2-D view (Q-13, Q-18).
    VIEW_POSITION_NAMES = False

    # The 75 label columns. harmonize() sorts them again.
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

    def _build_patient_id(self) -> None:
        # The release has no patient column. A study is a patient, so a patient split keeps
        # all series of a study together.
        self.df["patient_id"] = self.df["study_id"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["study_id"].astype(str)

    def _build_image_path(self) -> None:
        # The series folder. ITKReader loads it as a 3-D volume.
        self.df["image_path"] = (
            self.df["study_id"].astype(str) + "/" + self.df["series_id"].astype(str)
        )

    def _build_bbox(self) -> None:
        """Convert the points of ``train_label_coordinates.csv`` to tiny cube boxes.

        A point has ``(study_id, series_id, instance_number, condition, level, x, y)``. The
        box axis 0 is the slice axis in **MONAI dim 2 order**, the order in which ITKReader
        stacks the slices. After :func:`_base_load_3d` runs ``_SITKOrientD`` and
        ``Transposed([0,3,2,1])``, the box fits the reoriented image.

        * ``rank_in_loader_order(instance_number) / n_instances``: axis 0 (D, slice)
        * ``y / Rows``: axis 1 (H)
        * ``x / Columns``: axis 2 (W)

        The rank comes from :func:`_loader_file_order`, the same file list that ITKReader
        stacks. Ascending ``ImagePositionPatient[2]`` and ``InstanceNumber`` both fail for
        some series (V-H3-7).

        A row with no points, or a series with no readable DICOM, gets empty lists. A point
        that gives no box adds to one warning with the count (D-07, V-H3-33). The causes are a
        series that cannot be read, or a slice outside the first GDCM sub-series that the
        loader reads. At the end, ``clip_boxes`` clips each box to [0, 1] (D-07).
        """
        if self.coord_csv_path is None:
            # Without the CSV, the table has no box columns (as in v0).
            return

        if not self.base_image_dir or not os.path.isdir(self.base_image_dir):
            raise ValueError(
                "base_image_dir must point at an existing directory when "
                "coord_csv_path is set (DICOM headers are needed for "
                "bbox normalisation)."
            )

        coords = pd.read_csv(self.coord_csv_path)
        # Kaggle gives the IDs as integers.
        coords["study_id"] = coords["study_id"].astype(str)
        coords["series_id"] = coords["series_id"].astype(str)

        # For each series, list the files in the slice order of the 3-D loader. A
        # ``{InstanceNumber: rank}`` map then gives the MONAI dim 2 index of a slice.
        import itk
        import pydicom

        series_file_names = itk.GDCMSeriesFileNames  # get it in the main thread
        bad_files: list[str] = []  # slice files with no readable header

        def _probe_series(
            rel_path: str,
        ) -> tuple[str, tuple[dict[int, int], int, int, int] | None]:
            """Return (rel_path, (rank_of_inst, n_instances, Rows, Columns)) or None."""
            full = os.path.join(self.base_image_dir, rel_path)
            try:
                files = _loader_file_order(full, series_file_names)
                if not files:
                    return rel_path, None
                rows = cols = None
                rank_of_inst: dict[int, int] = {}
                for rank, path in enumerate(files):
                    try:
                        ds = pydicom.dcmread(
                            path,
                            stop_before_pixels=True,
                            specific_tags=["InstanceNumber", "Rows", "Columns"],
                        )
                        if rows is None:
                            rows, cols = int(ds.Rows), int(ds.Columns)
                        rank_of_inst[int(ds.InstanceNumber)] = rank
                    except Exception:
                        # The slice gets no rank, so its points count as lost (warning below).
                        bad_files.append(path)
                if not rank_of_inst or rows is None:
                    return rel_path, None
                return rel_path, (rank_of_inst, len(files), rows, cols)
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

        coord_groups: dict[tuple[str, str], pd.DataFrame] = dict(
            tuple(coords.groupby(["study_id", "series_id"]))
        )

        eps = self.bbox_epsilon
        eps_z_user = self.bbox_z_epsilon
        # Points with no box, for the D-07 warning (R-code-20, V-H3-33).
        unread_series: list[str] = []
        n_points = n_lost = n_off_volume = 0

        def _row_bboxes(row) -> tuple[list, list]:
            nonlocal n_points, n_lost, n_off_volume
            key = (str(row["study_id"]), str(row["series_id"]))
            g = coord_groups.get(key)
            if g is None or g.empty:
                return [], []
            n_points += len(g)
            probe = probes.get(row["image_path"])
            if probe is None or min(probe[1:]) <= 0:
                unread_series.append(row["image_path"])
                n_lost += len(g)
                return [], []
            rank_of_inst, n_inst, H, W = probe
            eps_z = eps_z_user if eps_z_user is not None else 0.5 / n_inst
            boxes, labels = [], []
            for _, p in g.iterrows():
                inst = int(p["instance_number"])
                rank = rank_of_inst.get(inst)
                if rank is None:
                    # The slice is not in the first GDCM sub-series that the
                    # loader reads, or its header cannot be read (V-H3-33).
                    n_lost += 1
                    n_off_volume += 1
                    continue
                x_norm = p["x"] / W
                y_norm = p["y"] / H
                # Center of the slice in MONAI dim 2 order, from 0 to 1.
                z_norm = (rank + 0.5) / n_inst
                # clip_boxes clips the box to [0, 1] below (D-07).
                boxes.append([float(v) for v in (
                    z_norm - eps_z, z_norm + eps_z,
                    y_norm - eps, y_norm + eps,
                    x_norm - eps, x_norm + eps,
                )])
                labels.append(
                    (str(p["condition"]) + "_" + str(p["level"]))
                    .lower()
                    .replace(" ", "_")
                    .replace("/", "_")
                )
            return boxes, labels

        # apply() on a table with no rows gives a table, not a Series. With no rows (an id
        # filter), dtype=object keeps the list columns.
        bb_pairs = self.df.apply(_row_bboxes, axis=1) if len(self.df) else []
        self.df["bbox"] = pd.Series([b for b, _ in bb_pairs], index=self.df.index, dtype=object)
        self.df["bbox_labels"] = pd.Series([l for _, l in bb_pairs], index=self.df.index, dtype=object)
        if n_lost:
            msg = (
                f"RSNA 2024 lumbar spine: {n_lost} of {n_points} points give no box, because they are "
                "not on the loaded volume (D-07)."
            )
            if unread_series:
                msg += f" {len(set(unread_series))} series could not be read (first: {unread_series[0]})."
            if bad_files:
                msg += f" {len(bad_files)} slice files could not be read (first: {bad_files[0]})."
            if n_off_volume:
                msg += (f" {n_off_volume} points are on a second GDCM sub-series of a series folder, which "
                        "the 3-D loader does not read, or on a slice file that could not be read.")
            warnings.warn(msg, UserWarning, stacklevel=3)

        clip_boxes(self.df, "RSNA 2024 lumbar spine")  # D-07

    def _read_source(self) -> pd.DataFrame:
        """Join each study of ``train.csv`` with its series and expand the 25 targets to 75 0/1 columns."""
        if self.series_description_csv_path is None:
            raise ValueError(
                "series_description_csv_path is required for "
                "RSNA2024LumbarSpineHarmonizer (train_series_descriptions.csv)."
            )

        labels_df = pd.read_csv(self.csv_path)
        series_df = pd.read_csv(self.series_description_csv_path)

        # One row for each (study_id, series_id), with the 25 label columns. The inner join
        # drops a series whose study is not in train.csv.
        df = series_df.merge(labels_df, on="study_id", how="inner")

        # Each target gives 3 binary columns (normal_mild, moderate, severe). A blank severity
        # gives NaN in its 3 columns, not 0 (D-05, Q-17). This is 572 of 49,375 cells.
        for target in _BASE_TARGETS:
            vals = df[target]
            for severity, suffix in _SEVERITY_SUFFIX.items():
                df[f"{target}_{suffix}"] = (vals == severity).astype("float32").where(vals.notna())
        return df.drop(columns=_BASE_TARGETS)

    def verify_images(
        self,
        base_image_dir: str = None,
        drop_missing: bool = True,
    ) -> pd.DataFrame:
        """Check that each series folder exists and has at least one .dcm file.

        The base method checks files. Here ``image_path`` is a series folder.
        """
        df = self.harmonized_df
        # The folder rule of the base verify_images (D-04, V-H3-22).
        base_image_dir = base_image_dir or getattr(self, "base_image_dir", None)
        if not base_image_dir and not df["image_path"].astype(str).map(os.path.isabs).all():
            raise ValueError(
                "image_path is relative to base_image_dir (D-04). Pass "
                "verify_images(base_image_dir=...), or every row looks missing."
            )

        def _exists(p):
            fp = os.path.join(base_image_dir, p) if base_image_dir else p
            if not os.path.isdir(fp):
                return False
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
