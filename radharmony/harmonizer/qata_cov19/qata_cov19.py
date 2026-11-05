"""Harmonizer for QaTa-COV19 (v2): COVID-19 chest X-rays with infection masks.

The v2 release has 9,258 COVID-19 chest X-rays (7,145 train and 2,113 test). Each image has a
binary mask of the infected lung region. Source (the first author's own Kaggle upload):
https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset

The release has no CSV. The harmonizer lists the ``*.png`` files of ``base_dir`` and builds
all columns from the file names and the sibling ``Ground-truths/`` folder.
``base_dir`` is the ``Images/`` folder of one split, for example
``QaTa-COV19/QaTa-COV19-v2/Train Set/Images/``. ``masks_dir`` is ``../Ground-truths/`` by
default.

There are two file name schemes. ``covid_<n>.png`` (2,951 train files) has no subject,
session or view, so ``patient_id`` is the file stem. ``sub-<subject>_ses-<session>...``
(BIDS style, 4,194 train and 2,113 test files) has a subject, a session, a view (optional)
and a modality code. The harmonizer records the scheme in ``filename_scheme`` and does not
say where the images came from.

The subjects of train (1,656) and test (474) do not overlap. But test ``sub-S09512`` is the
same picture as train ``covid_4134``. In train, 37 pairs of files are near copies under 2
names. They are not pixel-equal (correlation above 0.99 at 32 x 32). Each copy has its own
``patient_id``. Our own split (``get_datasets()``) can put the 2 copies in train and val
(D-10). The published split stays as it is (D-34).

Every image is COVID-positive, so ``covid`` is always 1. It gives ``output_cls`` a value and
combines with a negative-control dataset. Alone it is not a classification task.

Details: ``docs/wiki/reference/datasets/cxr-boxes-masks/qata_cov19.md``.
"""

import os
import re

import pandas as pd

from ..base import BaseHarmonizer


#: Tokens of a BIDS-style stem, for example sub-S09810_ses-E16816_acq-1_run-1_bp-chest_vp-pa_dx.
#: Separate patterns, because the acq- and vp- tokens are missing on some images. One pattern
#: would lose the subject ID too.
_VIEW_RE = re.compile(r"_vp-([a-z]+)", re.IGNORECASE)
_ACQ_RE = re.compile(r"_(cr|dx)$", re.IGNORECASE)
_SUBJECT_RE = re.compile(r"^sub-([A-Za-z0-9]+)")
_SESSION_RE = re.compile(r"_ses-([A-Za-z0-9]+)")

#: Prefix of each mask file name.
_MASK_PREFIX = "mask_"


def _parse_stem(stem: str) -> dict:
    """Get the subject, session, view and acquisition from one file stem.

    ``filename_scheme`` is "bids" only for the ``sub-`` scheme. The ``covid_<n>`` scheme has no
    metadata, so all other fields are ``None`` and the caller uses the stem.
    """
    if not stem.startswith("sub-"):
        return {
            "filename_scheme": "qata",
            "subject": None,
            "session": None,
            "view_position": None,
            "acquisition": None,
        }

    m_sub = _SUBJECT_RE.match(stem)
    m_ses = _SESSION_RE.search(stem)
    m_view = _VIEW_RE.search(stem)
    m_acq = _ACQ_RE.search(stem)
    return {
        "filename_scheme": "bids",
        "subject": m_sub.group(1) if m_sub else None,
        "session": m_ses.group(1) if m_ses else None,
        # vp-ap, vp-pa and vp-ll give AP, PA and LL (the base step makes LL LATERAL, Q-13). 167
        # train and 179 test images have no view. They stay NaN.
        "view_position": m_view.group(1).upper() if m_view else None,
        # DICOM modality code at the end of the name: CR (computed) or DX (digital radiography).
        "acquisition": m_acq.group(1).upper() if m_acq else None,
    }


class QaTaCOV19Harmonizer(BaseHarmonizer):
    """Harmonize one QaTa-COV19-v2 split into the standard RadHarmony format.

    One class serves both splits. They differ only in ``base_dir``.

    Args:
        base_dir: The ``Images/`` folder of a split, for example
            ``.../QaTa-COV19/QaTa-COV19-v2/Train Set/Images/``.
        masks_dir: The ``Ground-truths/`` folder of the split. The default is
            ``../Ground-truths/`` next to ``base_dir``, as in the original download.
    """

    LABEL_COLS = ["covid"]

    LABEL_JOIN_COLS = None
    VIEW_POSITION_SOURCE_COL = None
    VIEW_POSITION_JOIN_COLS = None
    MASK_SOURCE_COL = None
    MASK_JOIN_COLS = None
    BBOX_SOURCE_COL = None
    BBOX_JOIN_COLS = None
    REPORT_JOIN_COLS = None
    REPORT_PATH_COL = None

    EXTRA_OUTPUT_COLS = ["filename_scheme", "acquisition"]

    def __init__(self, base_dir: str, masks_dir: str = None):
        self.base_dir = os.path.expanduser(base_dir)
        self.masks_dir = (
            os.path.expanduser(masks_dir)
            if masks_dir
            else os.path.normpath(os.path.join(self.base_dir, "..", "Ground-truths"))
        )
        super().__init__(csv_path=None)

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()
        snap["base_dir"] = self.base_dir
        snap["masks_dir"] = self.masks_dir
        snap.pop("csv_path", None)  # not an argument of this class (V-H2-1)
        return snap

    def _read_source(self) -> pd.DataFrame:
        """List the PNG files of ``base_dir`` and parse each file stem (one row for each image)."""
        if not os.path.isdir(self.base_dir):
            raise FileNotFoundError(f"Image directory not found: {self.base_dir}")

        stems = sorted(
            os.path.splitext(f)[0]
            for f in os.listdir(self.base_dir)
            if f.lower().endswith(".png")
        )
        if not stems:
            raise FileNotFoundError(
                f"No PNG images found in {self.base_dir}. base_dir must be a "
                f"QaTa-COV19-v2 split's Images/ directory, e.g. "
                f"'.../QaTa-COV19-v2/Train Set/Images/'."
            )

        parsed = pd.DataFrame([_parse_stem(s) for s in stems])
        df = pd.DataFrame({"_stem": stems})
        df["_subject"] = parsed["subject"].values
        df["_session"] = parsed["session"].values
        df["filename_scheme"] = parsed["filename_scheme"].values
        df["view_position"] = parsed["view_position"].values
        df["acquisition"] = parsed["acquisition"].values
        return df

    def _build_patient_id(self) -> None:
        # The QaTa scheme has no patient ID. The stem is a pseudo-patient, one for each image.
        # Filter on filename_scheme before a patient split.
        self.df["patient_id"] = self.df["_subject"].fillna(self.df["_stem"])

    def _build_study_id(self) -> None:
        self.df["study_id"] = self.df["_session"].fillna(self.df["patient_id"])

    def _build_image_path(self) -> None:
        self.df["image_path"] = self.df["_stem"] + ".png"

    def _build_labels(self) -> None:
        # Every image of v2 is COVID-positive.
        self.df["covid"] = 1
        self.df["covid"] = self.df["covid"].astype("Int64")

    def preprocess_masks(self, output_dir: str, base_image_dir: str = None, num_cores: int = 1) -> None:  # noqa: ARG002
        """Do nothing. The masks are PNG files already, so ``mask_path`` keeps the source files.

        The base step would decode ``mask_path`` as an encoded string and write empty masks.
        """

    def _build_mask_path(self) -> None:
        """Set ``mask_path`` to ``<masks_dir>/mask_<image file name>``.

        The path is absolute. ``get_data_dict`` adds ``base_image_dir`` to ``image_path`` only.
        A relative mask path would use the working folder. The masks are binary PNG files, so
        nothing needs a decode step.
        """
        if not os.path.isdir(self.masks_dir):
            self.df["mask_path"] = None
            return

        available = {
            f for f in os.listdir(self.masks_dir) if f.startswith(_MASK_PREFIX)
        }
        self.df["mask_path"] = self.df["_stem"].apply(
            lambda s: (
                os.path.join(self.masks_dir, _MASK_PREFIX + s + ".png")
                if _MASK_PREFIX + s + ".png" in available
                else None
            )
        )
