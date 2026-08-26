"""Harmonizers for the Chest ImaGenome dataset.

Chest ImaGenome is an annotation layer built on top of MIMIC-CXR: it does not
ship any pixel data.  For every chest X-ray it provides up to ~36 anatomical
region **bounding boxes** and, per region, the radiographic **findings** present
there (scene-graph attributes).  The images themselves come from the MIMIC-CXR
DICOM tree, keyed by ``dicom_id``.

**Source (PhysioNet - requires MIMIC credentialed access + signed DUA):**
  https://physionet.org/content/chest-imagenome/1.0.0/

**Cite:**
  Wu JT, et al. Chest ImaGenome Dataset for Clinical Reasoning. NeurIPS 2021
  (Datasets and Benchmarks).

Two releases share this module:

* **gold** (:class:`ChestImaGenomeGoldHarmonizer`) - 1,000 manually verified
  images with merged ground-truth boxes.
* **silver** (:class:`ChestImaGenomeSilverHarmonizer`) - the full auto-generated
  scene-graph set with official train/valid/test splits.

Each row of the harmonized DataFrame carries three parallel, index-aligned
per-box Python lists (matching the VinDr-CXR harmonizer):

* ``bbox``         - ``[y_min, y_max, x_min, x_max]`` normalized to ``[0, 1]``
  (dim0/dim1 order, matching the RadHarmony bbox pipeline convention).
* ``bbox_labels``  - the **anatomy** region name, e.g. ``"right lung"``.
* ``bbox_findings``- the **findings** for that region, ``|``-joined, e.g.
  ``"pneumonia|pleural effusion"`` (empty string when the region has none).

``bbox_findings`` is carried through :meth:`_select_harmonized_columns` via
``EXTRA_OUTPUT_COLS`` - the anatomy/findings split lives in the harmonized
DataFrame.  The dataset classes set ``SUPPORTS_BBOX_FINDINGS = True`` so that,
with ``output_bbox=True``, ``bbox_findings`` is also threaded into each sample
alongside ``bbox`` + ``bbox_labels`` (index-aligned through augmentation).

Bounding boxes are normalized against the original DICOM dimensions
(``Rows`` = H, ``Columns`` = W) read from the header, mirroring
:class:`radharmony.harmonizer.vindr_cxr.VinDrCXRTrainHarmonizer`.
"""

import json
import os
import zipfile

import pandas as pd

from ..base import BaseHarmonizer


# Clinically meaningful finding categories.  ``nlp`` (yes/normal/abnormal meta)
# and ``technicalassessment`` (image-quality) are excluded so ``bbox_findings``
# holds only real radiographic findings.
_FINDING_CATEGORIES = frozenset(
    {"anatomicalfinding", "disease", "tubesandlines", "device"}
)


class ChestImaGenomeHarmonizer(BaseHarmonizer):
    """Shared base for the Chest ImaGenome gold/silver harmonizers.

    Args:
        annotation_dir: Root of the Chest ImaGenome release (the directory that
            contains ``gold_dataset/``, ``silver_dataset/``, ``utils/``), e.g.
            ``/data/CHEST-IMAGENOME/``.
        base_image_dir: Root of the MIMIC-CXR **DICOM** file tree (the directory
            whose children are ``p10/``, ``p11/`` ...), e.g.
            ``/data/MIMIC-CXR-V2-AWS/files/``.  Used to read image dimensions
            for bbox normalization; the harmonized ``image_path`` is relative to
            this directory.
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

    # Per-box findings + image dims are carried through in addition to the
    # standard optional columns (bbox / bbox_labels / view_position).
    EXTRA_OUTPUT_COLS = ["bbox_findings", "image_width", "image_height"]

    def __init__(self, annotation_dir: str, base_image_dir: str = None):
        self.annotation_dir = (
            os.path.expanduser(annotation_dir) if annotation_dir else None
        )
        self.base_image_dir = (
            os.path.expanduser(base_image_dir) if base_image_dir else None
        )
        # csv_path is unused (harmonize() is fully overridden); keep the base
        # attribute populated for consistency.
        super().__init__(csv_path=self.annotation_dir)
        self._dim_cache: dict = {}

    def _harmonizer_init_snapshot(self) -> dict:
        # Fully overridden: the base snapshot injects csv_path, which this
        # harmonizer's __init__ does not accept.
        return {
            "annotation_dir": self.annotation_dir,
            "base_image_dir": self.base_image_dir,
        }

    # -- abstract stubs (harmonize() is fully overridden by subclasses) --
    def _build_patient_id(self) -> None:  # pragma: no cover - unused
        pass

    def _build_study_id(self) -> None:  # pragma: no cover - unused
        pass

    def _build_image_path(self) -> None:  # pragma: no cover - unused
        pass

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _read_dims(self, rel_dcm_path: str):
        """Return ``(rows_H, cols_W)`` from the DICOM header (cached).

        Returns ``(None, None)`` when ``base_image_dir`` is unset or the file
        cannot be read.  Only the ``Rows``/``Columns`` tags are decoded, so this
        stays cheap even across large silver splits.
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
    def _norm_box(x1, y1, x2, y2, H: int, W: int) -> list:
        """Original-pixel ``[x1,y1,x2,y2]`` -> normalized ``[y_min,y_max,x_min,x_max]``.

        Normalized to ``[0, 1]`` against the original image dims (H=Rows,
        W=Columns), the dim0/dim1 order expected by the RadHarmony bbox
        pipeline (see ``_BboxToMask`` in ``dataset/transforms.py``).
        """
        ymin, ymax = sorted((float(y1) / H, float(y2) / H))
        xmin, xmax = sorted((float(x1) / W, float(x2) / W))

        def _clip(v: float) -> float:
            return max(0.0, min(1.0, v))

        return [_clip(ymin), _clip(ymax), _clip(xmin), _clip(xmax)]

    @staticmethod
    def _findings_from_tokens(token_lists) -> list:
        """Collect real findings from scene-graph attribute token lists.

        ``token_lists`` is a list of lists of ``"category|relation|finding"``
        strings.  Keeps ``relation == "yes"`` tokens in the clinically
        meaningful categories; returns a sorted unique list of finding names.
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

        ``bbox`` / ``bbox_labels`` / ``bbox_findings`` are stored as parallel
        Python lists (matching the VinDr-CXR harmonizer), so per-sample outputs
        are list-typed and the harmonizer's ``save()`` (pickle) round-trips them.
        """
        return {
            "patient_id": str(meta_subject),
            "study_id": str(meta_study),
            "image_path": rel_path,
            "view_position": view,
            "image_width": int(W),
            "image_height": int(H),
            "bbox": bbox,
            "bbox_labels": labels,
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
    """

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
        ann = self.annotation_dir
        gold = os.path.join(ann, "gold_dataset")

        boxes = pd.read_csv(
            os.path.join(gold, "gold_bbox_coordinate_annotations_1000images.csv")
        )
        boxes["dicom_id"] = boxes["image_id"].str.replace(".dcm", "", regex=False)

        # --- per-(image, region) positive findings ---
        attr = pd.read_csv(
            os.path.join(gold, "gold_object_attribute_with_coordinates.txt"),
            sep="\t",
        )
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

        # --- dicom_id -> subject / study / path / view ---
        rl = pd.read_csv(os.path.join(ann, "utils", "cxr-record-list_view.csv"))
        rl = rl.drop_duplicates("dicom_id").set_index("dicom_id")

        rows = []
        for did, g in boxes.groupby("dicom_id", sort=False):
            if did not in rl.index:
                continue
            meta = rl.loc[did]
            rel = str(meta["path"]).removeprefix("files/")  # p10/.../<did>.dcm
            H, W = self._read_dims(rel)
            if not H or not W:
                continue
            bbox, labels, findings = [], [], []
            for _, r in g.iterrows():
                region = r["bbox_name"]
                bbox.append(
                    self._norm_box(
                        r["original_x1"], r["original_y1"],
                        r["original_x2"], r["original_y2"], H, W,
                    )
                )
                labels.append(region)
                findings.append("|".join(find_map.get((did, region), [])))
            rows.append(
                self._pack_row(
                    meta["subject_id"], meta["study_id"], rel,
                    meta.get("ViewPosition"), H, W, bbox, labels, findings,
                )
            )

        self.df = pd.DataFrame(rows).reset_index(drop=True)
        return self._select_harmonized_columns(self.df)


class ChestImaGenomeSilverHarmonizer(ChestImaGenomeHarmonizer):
    """Harmonize the Chest ImaGenome **silver** set (full scene-graph release).

    Parses ``silver_dataset/scene_graph.zip`` (one JSON per image: ``objects``
    = boxes, ``attributes`` = per-region findings) for the images listed in the
    selected split (``silver_dataset/splits/<split>.csv``), excluding
    ``images_to_avoid.csv``.

    Args:
        annotation_dir: Chest ImaGenome release root (see base class).
        base_image_dir: MIMIC-CXR DICOM ``files/`` root (see base class).
        split: One of ``"all"`` (default), ``"train"``, ``"valid"``, ``"test"``.
            ``"all"`` concatenates train+valid+test.
    """

    EXTRA_OUTPUT_COLS = (
        ChestImaGenomeHarmonizer.EXTRA_OUTPUT_COLS + ["split"]
    )

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

    def harmonize(self, *args, **kwargs) -> pd.DataFrame:
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
        for meta in split_df.itertuples(index=False):
            did = meta.dicom_id
            name = f"scene_graph/{did}_SceneGraph.json"
            if name not in available:
                continue
            sg = json.loads(zf.read(name))
            objs = sg.get("objects", [])
            if not objs:
                continue
            rel = str(meta.path).removeprefix("files/")
            H, W = self._read_dims(rel)
            if not H or not W:
                continue
            # region -> findings
            fmap = {
                a.get("bbox_name"): self._findings_from_tokens(a.get("attributes"))
                for a in sg.get("attributes", [])
            }
            bbox, labels, findings = [], [], []
            for o in objs:
                region = o["bbox_name"]
                bbox.append(
                    self._norm_box(
                        o["original_x1"], o["original_y1"],
                        o["original_x2"], o["original_y2"], H, W,
                    )
                )
                labels.append(region)
                findings.append("|".join(fmap.get(region, [])))
            row = self._pack_row(
                meta.subject_id, meta.study_id, rel,
                getattr(meta, "ViewPosition", None), H, W, bbox, labels, findings,
            )
            row["split"] = meta.split
            rows.append(row)

        self.df = pd.DataFrame(rows).reset_index(drop=True)
        return self._select_harmonized_columns(self.df)
