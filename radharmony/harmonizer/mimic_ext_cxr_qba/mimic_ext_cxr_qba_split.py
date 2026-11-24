"""Harmonizers for the official MIMIC-Ext-CXR-QBA train, validate and test splits (D-09)."""

from .mimic_ext_cxr_qba import MIMICExtCXRQBAHarmonizer


class MIMICExtCXRQBATrainHarmonizer(MIMICExtCXRQBAHarmonizer):
    """MIMIC-Ext-CXR-QBA questions of the train patients in ``patient_metadata.parquet``."""

    SPLIT = "train"


class MIMICExtCXRQBAValidateHarmonizer(MIMICExtCXRQBAHarmonizer):
    """MIMIC-Ext-CXR-QBA questions of the validate patients in ``patient_metadata.parquet``."""

    SPLIT = "validate"


class MIMICExtCXRQBATestHarmonizer(MIMICExtCXRQBAHarmonizer):
    """MIMIC-Ext-CXR-QBA questions of the test patients in ``patient_metadata.parquet``."""

    SPLIT = "test"
