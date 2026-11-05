"""Test-split harmonizer for the SIIM COVID-19 Detection Challenge."""

import pandas as pd

from .._dicom_tags import read_dicom_tags, tag_values
from .siim_covid19 import SIIMCOVID19Harmonizer


class SIIMCOVID19TestHarmonizer(SIIMCOVID19Harmonizer):
    """Test-split harmonizer for SIIM COVID-19 (images only, no labels).

    Walks the ``test/`` directory tree (``study/series/sop.dcm``) instead of
    reading the train CSVs.  No class labels or bounding boxes are produced.
    ``patient_id`` is the DICOM ``PatientID`` tag, or the study if the tag is missing.
    ``sex`` comes from the DICOM tag ``PatientSex`` (D-26).

    Args:
        base_image_dir: Root of the test DICOM tree
            (typically ``<competition_root>/test/``).
    """

    LABEL_COLS = []

    def __init__(self, base_image_dir: str, **kwargs):
        super().__init__(
            csv_path=None,
            image_csv_path=None,
            base_image_dir=base_image_dir,
        )

    def _read_source(self) -> pd.DataFrame:
        """Walk the test directory and return one row per image (no labels)."""
        sop_to_path = self._build_sop_to_path()
        if not sop_to_path:
            raise FileNotFoundError(
                f"No <study>/<series>/*.dcm file in {self.base_image_dir}. Check the path of the test image "
                "folder (D-35)."
            )
        paths = list(sop_to_path.values())
        headers = read_dicom_tags(self.base_image_dir, paths, ["PatientID", "PatientSex"],
                                  self.num_workers)
        rows = []
        for rel_path in paths:
            study = rel_path.split("/")[0]
            patient = (headers.get(rel_path) or {}).get("PatientID") or study
            rows.append({"patient_id": str(patient), "study_id": study, "image_path": rel_path})
        df = pd.DataFrame(rows)
        df["PatientSex"] = tag_values(headers, paths, "PatientSex")
        return df

    def _build_patient_id(self) -> None:
        pass  # _read_source() sets it

    def _build_bbox(self) -> None:
        pass  # no boxes
