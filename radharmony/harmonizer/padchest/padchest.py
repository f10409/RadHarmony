"""Harmonizer for the PadChest (BIMCV) dataset."""

import ast

import numpy as np
import pandas as pd

from .._age import to_years
from .._ids import warn_shared_study_ids
from ..base import BaseHarmonizer


class PadChestHarmonizer(BaseHarmonizer):
    """Harmonize PadChest into the standard RadHarmony format.

    PadChest (BIMCV) has 160,861 Spanish chest X-ray images of about 67,000 patients from
    Hospital San Juan de Alicante. The 193 labels come from NLP on the Spanish reports and
    a manual review. The ``Labels`` column holds Python list text, for example
    ``"['cardiomegaly', 'pleural effusion']"``. Some labels have a leading space
    (`` pneumonia``). The harmonizer strips it, so both forms give one column.

    The PNG files are in 52 subfolders of ``base_image_dir`` (``0/`` to ``50/`` and ``54/``).
    ``image_path`` has the subfolder name, as for ChestX-ray14.

    Args:
        csv_path: Path to ``PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz``.
        base_image_dir: Root folder with the subfolders ``0/`` to ``54/``, for example
            ``/data/PadChest/images/``.
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
    # ``Projection`` has short names (PA, L, AP, AP_horizontal, COSTAL). ``ViewPosition_DICOM``
    # has the raw names.
    VIEW_POSITION_SOURCE_COL = "Projection"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    #: ``PatientSex_DICOM``: M, F, O or empty. The base step makes ``sex`` (D-26).
    #: The CSV has no age, only ``PatientBirth`` (a year). See ``_build_age_years``.
    SEX_SOURCE_COL = "PatientSex_DICOM"

    #: Labels about the quality of the image, not a finding. With ``drop_uncertain=True``
    #: (default, as in CheXpert), :meth:`_build_labels` drops the rows with any of them.
    _UNCERTAIN_META_LABELS = ("exclude", "suboptimal_study")

    def __init__(self, csv_path: str, base_image_dir: str = None):
        super().__init__(csv_path=csv_path)
        self.base_image_dir = base_image_dir
        self.drop_uncertain = True  # harmonize() can change it

    def harmonize(self, *args, drop_uncertain: bool = True, **kwargs):
        """Harmonize PadChest.

        PadChest has no CheXpert-style ``-1`` mark. Two image labels flag bad data: ``exclude``
        (1,312 rows) and ``suboptimal_study`` (1,839 rows, about 3,151 rows in total after
        de-duplication). With ``drop_uncertain=True`` (default), the rows with either label are
        dropped. Use ``drop_uncertain=False`` to keep them.
        """
        self.drop_uncertain = drop_uncertain
        return super().harmonize(*args, **kwargs)

    def _build_age_years(self, df: pd.DataFrame) -> pd.DataFrame:
        """Give ``age_years`` = study year - ``PatientBirth`` (D-06).

        The age can be 1 year too high, because the CSV has no birth date. A
        missing birth year, or an age outside 0 to 130, gives NaN.
        """
        if not {"PatientBirth", "StudyDate_DICOM"} <= set(df.columns):
            return super()._build_age_years(df)
        study_year = pd.to_numeric(df["StudyDate_DICOM"], errors="coerce") // 10000
        df = df.copy()
        df["age_years"] = to_years(study_year - pd.to_numeric(df["PatientBirth"], errors="coerce"), "years")
        return df

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["PatientID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["StudyID"].astype(str)
        # The 2019 release gives 1 StudyID to 2 patients (V-H2-32).
        warn_shared_study_ids(self.df, "PadChest", "StudyID")

    def _build_image_path(self) -> None:
        # ImageDir is an integer in the CSV. The path is like "0/<hash>_<slug>.png".
        self.df["image_path"] = (
            self.df["ImageDir"].astype(str) + "/" + self.df["ImageID"].astype(str)
        )

    def _build_labels(self) -> None:
        """Parse the ``Labels`` list text and make one 0/1 column for each label.

        It strips the space of a label such as `` pneumonia``. It drops the rows with NaN or
        unreadable ``Labels`` (about 103 of 160,861).
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

        # One matrix and one concat, not 193 column inserts (those give a PerformanceWarning).
        label_idx = {col: i for i, col in enumerate(self.LABEL_COLS)}
        arr = np.zeros((len(parsed), len(self.LABEL_COLS)), dtype=np.float32)
        for row_i, labels_set in enumerate(parsed.values):
            if not labels_set:
                arr[row_i] = np.nan  # an empty list says nothing, so not 0 for all (D-05, Q-15)
                continue
            for lbl in labels_set:
                col_i = label_idx.get(lbl)
                if col_i is not None:
                    arr[row_i, col_i] = 1
        label_df = pd.DataFrame(arr, columns=self.LABEL_COLS, index=self.df.index)
        self.df = pd.concat([self.df, label_df], axis=1)

        # The base class renames only the labels of a separate CSV.
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
        """Copy the ``Report`` column to ``report``.

        The reports are Spanish text, stemmed, inside the CSV. There is no report file, so the
        base method is not used. NaN stays NaN.
        """
        if "Report" in self.df.columns:
            self.df["report"] = self.df["Report"]
