"""Test-split harmonizer for the RSNA PE Detection Challenge (2020)."""

from .rsna_pe_detection import RSNAPEDetectionHarmonizer


class RSNAPEDetectionTestHarmonizer(RSNAPEDetectionHarmonizer):
    """Test-split harmonizer for RSNA PE Detection.

    Reads ``test.csv`` (columns ``StudyInstanceUID``, ``SeriesInstanceUID``,
    ``SOPInstanceUID``) and keeps one row per study. ``image_path`` is ``StudyUID/SeriesUID``.
    There are no labels.

    Args:
        csv_path: Path to ``test.csv``.
        base_image_dir: Root of the test DICOM tree, usually ``<competition_root>/test/``.
    """

    LABEL_COLS = []
