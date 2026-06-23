"""Harmonizer for the PadChest (BIMCV) dataset."""

import ast

import numpy as np
import pandas as pd

from .base import BaseHarmonizer


class PadChestHarmonizer(BaseHarmonizer):
    """Harmonize PadChest into the standard RadHarmony format.

    PadChest (BIMCV) is a Spanish chest-X-ray dataset of 160,861 images from
    ~67,000 patients at Hospital San Juan de Alicante. Labels (193 distinct
    findings) are NLP-derived from Spanish radiology reports plus manual
    review and stored as Python literal string lists in the ``Labels``
    column (e.g. ``"['cardiomegaly', 'pleural effusion']"``). Some raw
    labels have leading-space variants (`` pneumonia`` vs ``pneumonia``);
    per-label whitespace is stripped so both merge into the same one-hot
    column.

    Images are PNG files distributed across 52 sibling subdirectories
    under ``base_image_dir`` (e.g. ``.../PadChest/images/``):
    ``0/``, ``1/``, …, ``50/``, plus ``54/`` (slots 51/52/53 don't exist).
    This is the same structural-exception pattern as ChestX-ray14's
    ``images_001/…/images_012/`` — ``base_image_dir`` is the deepest stable
    directory and ``image_path`` carries the subdir name.

    Args:
        csv_path: Path to
            ``PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz``.
        base_image_dir: Root directory containing ``0/``, ``1/``, …, ``54/``
            subdirectories (e.g. ``/mnt/.../PadChest/images/``).
    """

    LABEL_COLS = [
        "COPD signs",
        "Chilaiditi sign",
        "NSG tube",
        "abnormal foreign body",
        "abscess",
        "adenopathy",
        "air bronchogram",
        "air fluid level",
        "air trapping",
        "alveolar pattern",
        "aortic aneurysm",
        "aortic atheromatosis",
        "aortic button enlargement",
        "aortic elongation",
        "aortic endoprosthesis",
        "apical pleural thickening",
        "artificial aortic heart valve",
        "artificial heart valve",
        "artificial mitral heart valve",
        "asbestosis signs",
        "ascendent aortic elongation",
        "atelectasis",
        "atelectasis basal",
        "atypical pneumonia",
        "axial hyperostosis",
        "azygoesophageal recess shift",
        "azygos lobe",
        "blastic bone lesion",
        "bone cement",
        "bone metastasis",
        "breast mass",
        "bronchiectasis",
        "bronchovascular markings",
        "bullas",
        "calcified adenopathy",
        "calcified densities",
        "calcified fibroadenoma",
        "calcified granuloma",
        "calcified mediastinal adenopathy",
        "calcified pleural plaques",
        "calcified pleural thickening",
        "callus rib fracture",
        "cardiomegaly",
        "catheter",
        "cavitation",
        "central vascular redistribution",
        "central venous catheter",
        "central venous catheter via jugular vein",
        "central venous catheter via subclavian vein",
        "central venous catheter via umbilical vein",
        "cervical rib",
        "chest drain tube",
        "chronic changes",
        "clavicle fracture",
        "consolidation",
        "costochondral junction hypertrophy",
        "costophrenic angle blunting",
        "cyst",
        "dai",
        "descendent aortic elongation",
        "dextrocardia",
        "diaphragmatic eventration",
        "double J stent",
        "dual chamber device",
        "electrical device",
        "emphysema",
        "empyema",
        "end on vessel",
        "endoprosthesis",
        "endotracheal tube",
        "esophagic dilatation",
        "exclude",
        "external foreign body",
        "fibrotic band",
        "fissure thickening",
        "flattened diaphragm",
        "fracture",
        "gastrostomy tube",
        "goiter",
        "granuloma",
        "ground glass pattern",
        "gynecomastia",
        "heart insufficiency",
        "heart valve calcified",
        "hemidiaphragm elevation",
        "hiatal hernia",
        "hilar congestion",
        "hilar enlargement",
        "humeral fracture",
        "humeral prosthesis",
        "hydropneumothorax",
        "hyperinflated lung",
        "hypoexpansion",
        "hypoexpansion basal",
        "increased density",
        "infiltrates",
        "interstitial pattern",
        "kerley lines",
        "kyphosis",
        "laminar atelectasis",
        "lepidic adenocarcinoma",
        "lipomatosis",
        "lobar atelectasis",
        "loculated fissural effusion",
        "loculated pleural effusion",
        "lung metastasis",
        "lung vascular paucity",
        "lymphangitis carcinomatosa",
        "lytic bone lesion",
        "major fissure thickening",
        "mammary prosthesis",
        "mass",
        "mastectomy",
        "mediastinal enlargement",
        "mediastinal mass",
        "mediastinal shift",
        "mediastinic lipomatosis",
        "metal",
        "miliary opacities",
        "minor fissure thickening",
        "multiple nodules",
        "nephrostomy tube",
        "nipple shadow",
        "nodule",
        "non axial articular degenerative changes",
        "normal",
        "obesity",
        "osteopenia",
        "osteoporosis",
        "osteosynthesis material",
        "pacemaker",
        "pectum carinatum",
        "pectum excavatum",
        "pericardial effusion",
        "pleural effusion",
        "pleural mass",
        "pleural plaques",
        "pleural thickening",
        "pneumomediastinum",
        "pneumonia",
        "pneumoperitoneo",
        "pneumothorax",
        "post radiotherapy changes",
        "prosthesis",
        "pseudonodule",
        "pulmonary artery enlargement",
        "pulmonary artery hypertension",
        "pulmonary edema",
        "pulmonary fibrosis",
        "pulmonary hypertension",
        "pulmonary mass",
        "pulmonary venous hypertension",
        "reservoir central venous catheter",
        "respiratory distress",
        "reticular interstitial pattern",
        "reticulonodular interstitial pattern",
        "rib fracture",
        "right sided aortic arch",
        "round atelectasis",
        "sclerotic bone lesion",
        "scoliosis",
        "segmental atelectasis",
        "single chamber device",
        "soft tissue mass",
        "sternoclavicular junction hypertrophy",
        "sternotomy",
        "subacromial space narrowing",
        "subcutaneous emphysema",
        "suboptimal study",
        "superior mediastinal enlargement",
        "supra aortic elongation",
        "surgery",
        "surgery breast",
        "surgery heart",
        "surgery humeral",
        "surgery lung",
        "surgery neck",
        "suture material",
        "thoracic cage deformation",
        "total atelectasis",
        "tracheal shift",
        "tracheostomy tube",
        "tuberculosis",
        "tuberculosis sequelae",
        "unchanged",
        "vascular hilar enlargement",
        "vascular redistribution",
        "ventriculoperitoneal drain tube",
        "vertebral anterior compression",
        "vertebral compression",
        "vertebral degenerative changes",
        "vertebral fracture",
        "volume loss",
    ]

    LABEL_JOIN_COLS = None
    # The "Projection" column is cleaner than ViewPosition_DICOM (PA/L/AP/
    # AP_horizontal/COSTAL vs the raw POSTEROANTERIOR/LATERAL/...).
    VIEW_POSITION_SOURCE_COL = "Projection"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    #: Meta-labels that signal annotation quality / image-level rejection
    #: rather than a clinical finding. When ``drop_uncertain=True`` (default,
    #: mirroring the CheXpert pattern), rows positive for any of these are
    #: dropped at the end of :meth:`_build_labels`.
    _UNCERTAIN_META_LABELS = ("exclude", "suboptimal_study")

    def __init__(self, csv_path: str, base_image_dir: str = None):
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir
        self.drop_uncertain = True  # default; overridden by harmonize() kwarg

    def harmonize(self, *args, drop_uncertain: bool = True, **kwargs):
        """Harmonize PadChest.

        PadChest has no CheXpert-style ``-1`` per-label uncertainty marker.
        The closest analog is two image-level meta-labels that the
        annotators used to flag bad data: ``exclude`` (1,312 rows) and
        ``suboptimal_study`` (1,839 rows; ~3,151 rows total after dedup).
        When ``drop_uncertain=True`` (default), rows positive for either
        are removed.  Pass ``drop_uncertain=False`` to keep them — useful
        when training on the full noisy distribution.
        """
        self.drop_uncertain = drop_uncertain
        return super().harmonize(*args, **kwargs)

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["PatientID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyID"].astype(str)

    def _build_image_path(self) -> None:
        # ImageDir is stored as int in the CSV (0..50, 54); cast to str so
        # the resulting relative path is e.g. "0/<hash>_<slug>.png".
        self.df["image_path"] = (
            self.df["ImageDir"].astype(str) + "/" + self.df["ImageID"].astype(str)
        )

    def _build_labels(self) -> None:
        """Parse the string-encoded Labels list and one-hot encode.

        Labels are stored as Python literal lists in the CSV (e.g.
        ``"['cardiomegaly', 'pleural effusion']"``). Some raw labels have
        leading-space variants (`` pneumonia`` vs ``pneumonia``); per-label
        whitespace is stripped so both merge.  Rows with NaN / unparseable
        Labels (~103 of 160,861) are dropped per the config's
        ``uncertain_handling.drop_uncertain_rows = true``.
        """

        def _parse(s):
            if pd.isna(s):
                return None
            try:
                items = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return None
            return frozenset(
                x.strip() for x in items if isinstance(x, str) and x.strip()
            )

        parsed = self.df["Labels"].apply(_parse)
        keep = parsed.notna()
        self.df = self.df[keep].copy()
        parsed = parsed[keep]

        # Single-pass one-hot: build the (n_rows, 193) int8 matrix from the
        # parsed label sets, then concat as one block. Avoids the per-column
        # frame.insert cost (and the resulting PerformanceWarning) of writing
        # 193 columns one at a time.
        label_idx = {col: i for i, col in enumerate(self.LABEL_COLS)}
        arr = np.zeros((len(parsed), len(self.LABEL_COLS)), dtype=np.int8)
        for row_i, labels_set in enumerate(parsed.values):
            for lbl in labels_set:
                col_i = label_idx.get(lbl)
                if col_i is not None:
                    arr[row_i, col_i] = 1
        label_df = pd.DataFrame(arr, columns=self.LABEL_COLS, index=self.df.index)
        self.df = pd.concat([self.df, label_df], axis=1)

        # Inline labels: snake-case rename here (base class only does this
        # for separate-CSV labels).
        for col in self.LABEL_COLS:
            snake = col.lower().replace(" ", "_")
            if snake != col:
                self.df.rename(columns={col: snake}, inplace=True)

        if self.drop_uncertain:
            mask = pd.Series(False, index=self.df.index)
            for meta in self._UNCERTAIN_META_LABELS:
                if meta in self.df.columns:
                    mask |= self.df[meta] == 1
            self.df = self.df[~mask].copy()

    def _build_report(self) -> None:
        """Copy the Spanish Report column directly into ``self.df["report"]``.

        PadChest reports are pre-processed (stemmed/lemmatized) Spanish text
        stored inline in the CSV — there is no separate report file to
        read, so the base class's file-path-based ``_build_report`` is
        bypassed.  NaN entries propagate as NaN.
        """
        if "Report" in self.df.columns:
            self.df["report"] = self.df["Report"]
