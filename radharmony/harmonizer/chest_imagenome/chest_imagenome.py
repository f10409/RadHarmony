"""Harmonizers for the Chest ImaGenome dataset.

Chest ImaGenome adds region boxes and findings to MIMIC-CXR. It has no
pixel data. The images come from the MIMIC-CXR DICOM tree, keyed by
``dicom_id``. Source: https://physionet.org/content/chest-imagenome/1.0.0/
(MIMIC credentialed access). Cite: Wu et al., NeurIPS 2021 Datasets and
Benchmarks.

Two releases share this module:

* gold (:class:`ChestImaGenomeGoldHarmonizer`): 1,000 verified images.
* silver (:class:`ChestImaGenomeSilverHarmonizer`): the full auto-generated
  set with official train, valid and test splits.

Each row has three parallel lists, one item per box (as in VinDr-CXR):

* ``bbox``: ``[y_min, y_max, x_min, x_max]``, normalized to ``[0, 1]``.
* ``bbox_labels``: anatomy region name in snake case, such as ``"right_lung"``.
* ``bbox_findings``: findings of the region, joined with ``|``. It is an
  empty string if the region has none.

``bbox_findings`` goes out through ``EXTRA_OUTPUT_COLS``. The dataset classes
set ``SUPPORTS_BBOX_FINDINGS = True``, so it also goes into each sample when
``output_bbox=True``. Boxes are normalized with the image size from the
DICOM header (``Rows`` = H, ``Columns`` = W).

More: ``docs/wiki/reference/datasets/cxr-boxes-masks/chest_imagenome.md``.
"""

import json
import os
import warnings
import zipfile

import pandas as pd

from .._bbox import box_label, clip_box
from ..base import BaseHarmonizer


# Real finding categories. ``nlp`` and ``technicalassessment`` (image
# quality) are left out, so ``bbox_findings`` has only radiographic findings.
_FINDING_CATEGORIES = frozenset(
    {"anatomicalfinding", "disease", "tubesandlines", "device"}
)


class ChestImaGenomeHarmonizer(BaseHarmonizer):
    """Shared base for the Chest ImaGenome gold/silver harmonizers.

    Args:
        annotation_dir: Root of the Chest ImaGenome release. It holds
            ``gold_dataset/``, ``silver_dataset/`` and ``utils/``.
        base_image_dir: Root of the MIMIC-CXR DICOM tree (``files/``, with
            ``p10/``, ``p11/`` ... inside). It gives the image size for box
            normalization. ``image_path`` is relative to it.
    """

    LABEL_COLS = []

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    # Extra output: per-box findings and the image size.
    EXTRA_OUTPUT_COLS = ["bbox_findings", "image_width", "image_height"]

    def __init__(self, annotation_dir: str, base_image_dir: str = None):
        self.annotation_dir = (
            os.path.expanduser(annotation_dir) if annotation_dir else None
        )
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else None
        )
        # _read_source() is overridden, so csv_path is not read. Set it anyway.
        super().__init__(csv_path=self.annotation_dir)
        self._dim_cache: dict = {}

    def _harmonizer_init_snapshot(self) -> dict:
        # Fully overridden: the base snapshot injects csv_path, which this
        # harmonizer's __init__ does not accept.
        return {
            "annotation_dir": self.annotation_dir,
            "base_image_dir": self.base_image_dir,
        }

    # The _read_source() of each subclass gives patient_id, study_id and image_path.
    def _build_patient_id(self) -> None:
        pass

    def _build_study_id(self) -> None:
        pass

    def _build_image_path(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _require_base_image_dir(self) -> None:
        """Raise ``ValueError`` when ``base_image_dir`` is not set (V-H1-24).

        The boxes are fractions of the image size, and the size comes from the
        DICOM header. Without the folder, every image was skipped, and the
        empty table gave a ``KeyError`` that did not name the cause.
        """
        if not self.base_image_dir:
            raise ValueError(
                f"{type(self).__name__} needs base_image_dir, the MIMIC-CXR DICOM files/ "
                "folder. It reads the image size from each DICOM header to scale the boxes."
            )

    def _read_dims(self, rel_dcm_path: str):
        """Return ``(rows_H, cols_W)`` from the DICOM header (cached).

        Returns ``(None, None)`` if ``base_image_dir`` is unset or the file
        cannot be read. It decodes only the ``Rows`` and ``Columns`` tags.
        """
        if rel_dcm_path in self._dim_cache:
            return self._dim_cache[rel_dcm_path]
        H = W = None
        if self.base_image_dir:
            fp = os.path.join(self.base_image_dir, rel_dcm_path)
            try:
                import pydicom

                dcm = pydicom.dcmread(
                    fp, stop_before_pixels=True, specific_tags=["Rows", "Columns"]
                )
                H, W = int(dcm.Rows), int(dcm.Columns)
            except Exception:
                H = W = None
        self._dim_cache[rel_dcm_path] = (H, W)
        return H, W

    @staticmethod
    def _no_size_message(subset: str, n: int) -> str:
        """Text of the D-07 warning for ``n`` images with no DICOM header with Rows and Columns."""
        return (f"Chest ImaGenome {subset}: {n} images have no DICOM header with Rows and Columns in base_image_dir, so "
                "their boxes cannot be fractions of the image. The table does not keep them.")

    @staticmethod
    def _norm_box(x1, y1, x2, y2, H: int, W: int) -> list | None:
        """Original-pixel ``[x1,y1,x2,y2]`` -> normalized ``[y_min,y_max,x_min,x_max]``.

        Normalized to ``[0, 1]`` against the original image dims (H=Rows,
        W=Columns), the dim0/dim1 order expected by the RadHarmony bbox
        pipeline (D-07). The box is clipped to the image. Returns ``None`` when
        no area is left, because D-07 needs each min less than its max.
        Examples: a source box with ``x1 == x2``, or a box past the image edge.
        Also returns ``None`` for an inverted box (``x1 > x2`` or ``y1 > y2``).
        D-07 drops it and does not swap the corners (Q-9): the 22 inverted gold
        boxes are not on their region.
        """
        if float(x1) > float(x2) or float(y1) > float(y2):
            return None
        return clip_box([float(y1) / H, float(y2) / H, float(x1) / W, float(x2) / W])

    @staticmethod
    def _warn(message: str) -> None:
        """Warn the caller of ``harmonize()``. Call it from ``_read_source()`` only."""
        warnings.warn(
            message,
            UserWarning,
            stacklevel=4,  # the caller of harmonize(), past _read_source() and harmonize()
        )

    @staticmethod
    def _dropped_message(n: int) -> str:
        """Text of the D-07 warning for ``n`` source boxes with no area in the image, or inverted."""
        return (f"Chest ImaGenome: dropped {n} boxes with no area inside the image or with a min "
                "greater than its max (with their labels and findings).")

    @staticmethod
    def _findings_from_tokens(token_lists) -> list:
        """Collect real findings from scene-graph attribute token lists.

        ``token_lists`` is a list of lists of ``"category|relation|finding"``
        strings. It keeps the ``relation == "yes"`` tokens of real finding
        categories and returns the sorted, unique finding names.
        """
        found = set()
        for lst in token_lists or []:
            for tok in lst:
                parts = tok.split("|")
                if (
                    len(parts) == 3
                    and parts[1] == "yes"
                    and parts[0] in _FINDING_CATEGORIES
                ):
                    found.add(parts[2])
        return sorted(found)

    @staticmethod
    def _pack_row(meta_subject, meta_study, rel_path, view, H, W,
                  bbox, labels, findings) -> dict:
        """Assemble one harmonized row.

        ``bbox``, ``bbox_labels`` and ``bbox_findings`` are parallel Python
        lists (as in VinDr-CXR). ``save()`` (pickle) keeps them as lists.
        """
        return {
            "patient_id": str(meta_subject),
            "study_id": str(meta_study),
            "image_path": rel_path,
            "view_position": view,
            "image_width": int(W),
            "image_height": int(H),
            "bbox": bbox,
            "bbox_labels": [box_label(lab) for lab in labels],
            "bbox_findings": findings,
        }


class ChestImaGenomeGoldHarmonizer(ChestImaGenomeHarmonizer):
    """Harmonize the Chest ImaGenome **gold** set (1,000 verified images).

    Reads the merged ground-truth boxes
    (``gold_dataset/gold_bbox_coordinate_annotations_1000images.csv``, one row
    per ``(image, region)``), joins per-region findings from
    ``gold_dataset/gold_object_attribute_with_coordinates.txt`` (relation
    ``"yes"``), and resolves ``dicom_id -> subject/study/path`` via
    ``utils/cxr-record-list_view.csv``.

    The attribute file covers only 500 of the 1,000 images. The column
    ``findings_annotated`` is True for these images. For the other images, an
    empty ``bbox_findings`` entry means "not annotated", not "no finding"
    (V-H1-14).

    A positive finding on a region with no gold box is not in the table, and
    ``harmonize()`` warns with the count (D-07, Q-8). The gold box file has
    only "upper mediastinum", so the 227 findings on "mediastinum" have no box.
    """

    EXTRA_OUTPUT_COLS = ChestImaGenomeHarmonizer.EXTRA_OUTPUT_COLS + ["findings_annotated"]

    def _read_source(self) -> pd.DataFrame:
        """Return one row per gold image, with its boxes, region labels and findings."""
        self._require_base_image_dir()
        ann = self.annotation_dir
        gold = os.path.join(ann, "gold_dataset")

        boxes = pd.read_csv(
            os.path.join(gold, "gold_bbox_coordinate_annotations_1000images.csv")
        )
        boxes["dicom_id"] = boxes["image_id"].str.replace(".dcm", "", regex=False)

        # Positive findings of each (image, region).
        attr = pd.read_csv(
            os.path.join(gold, "gold_object_attribute_with_coordinates.txt"),
            sep="\t",
        )
        annotated = set(attr["image_id"].str.replace(".dcm", "", regex=False))
        attr = attr[
            (attr["context"] == "yes")
            & (attr["categoryID"].isin(_FINDING_CATEGORIES))
        ].copy()
        attr["dicom_id"] = attr["image_id"].str.replace(".dcm", "", regex=False)
        find_map: dict = {}
        for (did, region), g in attr.groupby(["dicom_id", "bbox"]):
            find_map[(did, region)] = sorted(
                set(g["label_name"].dropna().astype(str))
            )
        box_keys = set(zip(boxes["dicom_id"], boxes["bbox_name"], strict=True))
        # Only images in the box file. An image with no box row is not a region problem.
        box_ids = set(boxes["dicom_id"])
        no_box = {k: v for k, v in find_map.items() if v and k[0] in box_ids and k not in box_keys}
        if no_box:
            self._warn(
                f"Chest ImaGenome gold: {sum(map(len, no_box.values()))} positive findings in "
                f"{len({d for d, _ in no_box})} images are on a region with no box "
                f"({', '.join(sorted({r for _, r in no_box}))}). The table does not keep them."
            )

        # Map dicom_id to subject, study, path and view.
        rl = pd.read_csv(os.path.join(ann, "utils", "cxr-record-list_view.csv"))
        rl = rl.drop_duplicates("dicom_id").set_index("dicom_id")

        rows = []
        n_dropped = n_no_size = 0
        for did, g in boxes.groupby("dicom_id", sort=False):
            if did not in rl.index:
                continue
            meta = rl.loc[did]
            rel = str(meta["path"]).removeprefix("files/")  # p10/.../<did>.dcm
            H, W = self._read_dims(rel)
            if not H or not W:
                n_no_size += 1
                continue
            bbox, labels, findings = [], [], []
            for _, r in g.iterrows():
                region = r["bbox_name"]
                box = self._norm_box(
                    r["original_x1"], r["original_y1"],
                    r["original_x2"], r["original_y2"], H, W,
                )
                if box is None:
                    n_dropped += 1
                    continue
                bbox.append(box)
                labels.append(region)
                findings.append("|".join(find_map.get((did, region), [])))
            row = self._pack_row(
                meta["subject_id"], meta["study_id"], rel,
                meta.get("ViewPosition"), H, W, bbox, labels, findings,
            )
            row["findings_annotated"] = did in annotated
            rows.append(row)

        if n_no_size:
            self._warn(self._no_size_message("gold", n_no_size))
        if n_dropped:
            self._warn(self._dropped_message(n_dropped))
        return pd.DataFrame(rows)


class ChestImaGenomeSilverHarmonizer(ChestImaGenomeHarmonizer):
    """Harmonize the Chest ImaGenome **silver** set (full scene-graph release).

    Parses ``silver_dataset/scene_graph.zip`` (one JSON per image: ``objects``
    = boxes, ``attributes`` = per-region findings) for the images listed in the
    selected split (``silver_dataset/splits/<split>.csv``), excluding
    ``images_to_avoid.csv``. An image of the split list with no scene graph
    file is not in the table, and ``harmonize()`` warns with the count
    (V-H1-37). The valid split has 2 such images.

    ``sex`` comes from the scene-graph ``gender`` (D-26). ``age_years`` is the
    low edge of the scene-graph ``age_decile`` group, for example 50 for
    ``"50-60"`` and 90 for ``">=90"`` (D-06). The release writes the group
    10-20 as ``"20-Oct"``, so it gives 10.

    Args:
        annotation_dir: Chest ImaGenome release root (see base class).
        base_image_dir: MIMIC-CXR DICOM ``files/`` root (see base class).
        split: One of ``"all"`` (default), ``"train"``, ``"valid"``, ``"test"``.
            ``"all"`` concatenates train+valid+test.
    """

    EXTRA_OUTPUT_COLS = (
        ChestImaGenomeHarmonizer.EXTRA_OUTPUT_COLS + ["split"]
    )
    SEX_SOURCE_COL = "gender"
    # The low edge of the age group, made in _read_source() (D-06).
    AGE_SOURCE_COL = "age_decile_low"
    AGE_SOURCE_UNIT = "years"

    _VALID_SPLITS = ("all", "train", "valid", "test")

    def __init__(self, annotation_dir: str, base_image_dir: str = None,
                 split: str = "all"):
        if split not in self._VALID_SPLITS:
            raise ValueError(
                f"split must be one of {self._VALID_SPLITS}, got {split!r}."
            )
        super().__init__(annotation_dir=annotation_dir, base_image_dir=base_image_dir)
        self.split = split

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["split"] = self.split
        return snap

    @staticmethod
    def _age_group_low_edge(group) -> float:
        """Return the low edge of a scene-graph ``age_decile`` group, in years, or NaN.

        Example: ``"50-60"`` gives 50, ``">=90"`` gives 90. ``"20-Oct"`` is the
        group 10-20 after a spreadsheet read it as a date, so it gives 10.
        """
        if not isinstance(group, str):
            return float("nan")
        group = group.strip()
        if group == "20-Oct":
            return 10.0
        low = group.removeprefix(">=").split("-")[0]
        return float(low) if low.isdigit() else float("nan")

    def _read_source(self) -> pd.DataFrame:
        """Return one row per image of the split, with its boxes, region labels and findings."""
        self._require_base_image_dir()
        ann = self.annotation_dir
        silver = os.path.join(ann, "silver_dataset")
        splits_dir = os.path.join(silver, "splits")

        parts = ["train", "valid", "test"] if self.split == "all" else [self.split]
        frames = []
        for p in parts:
            f = pd.read_csv(os.path.join(splits_dir, f"{p}.csv"))
            f["split"] = p
            frames.append(f)
        split_df = pd.concat(frames, ignore_index=True)

        avoid = set(
            pd.read_csv(os.path.join(splits_dir, "images_to_avoid.csv"))["dicom_id"]
        )
        split_df = split_df[~split_df["dicom_id"].isin(avoid)]

        zf = zipfile.ZipFile(os.path.join(silver, "scene_graph.zip"))
        available = set(zf.namelist())

        rows = []
        n_dropped = n_no_graph = n_no_size = 0
        for meta in split_df.itertuples(index=False):
            did = meta.dicom_id
            name = f"scene_graph/{did}_SceneGraph.json"
            if name not in available:
                n_no_graph += 1
                continue
            sg = json.loads(zf.read(name))
            objs = sg.get("objects", [])
            if not objs:
                continue
            rel = str(meta.path).removeprefix("files/")
            H, W = self._read_dims(rel)
            if not H or not W:
                n_no_size += 1
                continue
            # Findings of each region.
            fmap = {
                a.get("bbox_name"): self._findings_from_tokens(a.get("attributes"))
                for a in sg.get("attributes", [])
            }
            bbox, labels, findings = [], [], []
            for o in objs:
                region = o["bbox_name"]
                box = self._norm_box(
                    o["original_x1"], o["original_y1"],
                    o["original_x2"], o["original_y2"], H, W,
                )
                if box is None:
                    n_dropped += 1
                    continue
                bbox.append(box)
                labels.append(region)
                findings.append("|".join(fmap.get(region, [])))
            row = self._pack_row(
                meta.subject_id, meta.study_id, rel,
                getattr(meta, "ViewPosition", None), H, W, bbox, labels, findings,
            )
            row["split"] = meta.split
            row["gender"] = sg.get("gender")
            row["age_decile_low"] = self._age_group_low_edge(sg.get("age_decile"))
            rows.append(row)

        if n_no_graph:
            self._warn(
                f"Chest ImaGenome silver: {n_no_graph} images of the split list have no scene graph "
                "file in scene_graph.zip. The table does not keep them."
            )
        if n_no_size:
            self._warn(self._no_size_message("silver", n_no_size))
        if n_dropped:
            self._warn(self._dropped_message(n_dropped))
        return pd.DataFrame(rows)
