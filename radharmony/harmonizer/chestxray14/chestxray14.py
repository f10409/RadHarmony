"""Harmonizer for the NIH ChestX-ray14 dataset."""

import os

import pandas as pd

from ..base import BaseHarmonizer


class ChestXray14Harmonizer(BaseHarmonizer):
    """Harmonize ChestX-ray14 into the standard RadHarmony format.

    This class holds the official train and test images together and is not
    for benchmarks. Use :class:`ChestXray14TrainHarmonizer` and
    :class:`ChestXray14TestHarmonizer`.

    With ``bbox_csv_path``, it drops the 880 images that have boxes (111,240
    images are left). For the images with boxes, use
    :class:`ChestXray14BboxHarmonizer`. The split classes keep their official
    list instead (``SPLIT_LIST``).

    ChestX-ray14 has 112,120 frontal-view chest X-rays from 30,805 patients.
    Labels are NLP-derived from radiology reports and stored as pipe-separated
    strings in the ``Finding Labels`` column
    (for example ``"Atelectasis|Effusion"``).

    Images are PNG files distributed across 12 subdirectories
    (``images_001/images/`` … ``images_012/images/``) under ``base_image_dir``
    (for example ``.../NIH_CXR/CXR14/``).

    Note: the raw CSV has ``Pleural_Thickening`` with a "_" already. Its
    snake case form is ``pleural_thickening``, with one "_".

    Args:
        csv_path: Path to ``Data_Entry_2017.csv``.
        bbox_csv_path: Path to ``BBox_List_2017.csv``. Used to drop the images
            with boxes. The split classes do not use it.
        base_image_dir: Root directory containing ``images_*/images/`` folders
            (for example ``.../NIH_CXR/CXR14/``).
    """

    LABEL_COLS = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "No Finding",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax",
    ]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = "View Position"
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    # Data_Entry_2017.csv: "Patient Age" in years and "Patient Gender" (M or F).
    # 16 ages are above 130 (for example 413), so their age_years is NaN (D-06).
    AGE_SOURCE_COL = "Patient Age"
    AGE_SOURCE_UNIT = "years"
    SEX_SOURCE_COL = "Patient Gender"

    # Official split list under base_image_dir. None: no split filter.
    SPLIT_LIST = None

    def __init__(
        self, csv_path: str, bbox_csv_path: str = None, base_image_dir: str = None
    ):
        super().__init__(csv_path=csv_path, bbox_csv_path=bbox_csv_path)
        self.base_image_dir = base_image_dir

    def _build_patient_id(self) -> None:
        self.df["patient_id"] = self.df["Patient ID"].astype(str)

    def _build_study_id(self) -> None:
        self.df["study_id"] = (
            self.df["Patient ID"].astype(str) + "_" + self.df["Follow-up #"].astype(str)
        )

    def _build_image_path(self) -> None:
        """Resolve image filenames to relative paths by scanning subdirectories.

        The CSV ``Image Index`` column contains just the filename
        (for example ``00000001_000.png``), but images live in
        ``images_001/images/`` … ``images_012/images/``.  We scan once to
        build a filename→relative-path lookup.
        """
        if self.base_image_dir is None:
            self.df["image_path"] = self.df["Image Index"]
            return

        lookup = {}
        for entry in sorted(os.listdir(self.base_image_dir)):
            subdir = os.path.join(self.base_image_dir, entry, "images")
            if os.path.isdir(subdir):
                for fname in os.listdir(subdir):
                    lookup[fname] = os.path.join(entry, "images", fname)

        self.df["image_path"] = self.df["Image Index"].map(lookup)

    def _build_labels(self) -> None:
        """One-hot encode the pipe-separated ``Finding Labels`` column."""
        for label in self.LABEL_COLS:
            self.df[label] = (
                self.df["Finding Labels"]
                .str.contains(
                    label.replace("_", "[_ ]"),
                    case=True,
                    regex=True,
                )
                .astype(int)
            )

        for col in self.LABEL_COLS:
            snake = col.lower().replace(" ", "_")
            if snake != col:
                self.df.rename(columns={col: snake}, inplace=True)

    def _read_source(self) -> pd.DataFrame:
        """Read ``Data_Entry_2017.csv`` and keep the rows of this class.

        A split class (``SPLIT_LIST`` set) keeps only the images of its official
        list, including the images with boxes. Otherwise, if you give
        ``bbox_csv_path``, it drops the images with boxes (they belong to
        :class:`ChestXray14BboxHarmonizer`).
        """
        df = super()._read_source()
        if self.SPLIT_LIST is not None:
            return self._keep_split_list(df)
        if self.bbox_csv_path is not None:
            bbox_df = pd.read_csv(
                self.bbox_csv_path,
                header=0,
                names=[
                    "image_id", "finding_label", "x", "y", "w", "h",
                    "_extra1", "_extra2", "_extra3",
                ],
                usecols=["image_id"],
            )
            df = df[~df["Image Index"].isin(set(bbox_df["image_id"].unique()))]
        return df

    def _build_bbox(self) -> None:
        # bbox_csv_path only drops the images with boxes (see _read_source). This class has no boxes.
        pass

    def _keep_split_list(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the rows of ``df`` whose image is in ``SPLIT_LIST``.

        Raises:
            ValueError: ``base_image_dir`` is None, so the list cannot be found.
            FileNotFoundError: the list is not in ``base_image_dir``.
        """
        if self.base_image_dir is None:
            raise ValueError(
                f"{type(self).__name__} needs base_image_dir to find {self.SPLIT_LIST}. "
                "Without the list, the table would hold the train and test images together."
            )
        list_path = os.path.join(self.base_image_dir, self.SPLIT_LIST)
        if not os.path.isfile(list_path):
            raise FileNotFoundError(
                f"{self.SPLIT_LIST!r} is not in {self.base_image_dir!r}. NIH ships it with the "
                "dataset. Without it, the table would hold the train and test images together."
            )
        with open(list_path) as f:
            allowed = set(f.read().splitlines())
        return df[df["Image Index"].isin(allowed)]
