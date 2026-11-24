"""Train-split harmonizer for the RSNA PE Detection Challenge (2020)."""

from .rsna_pe_detection import RSNAPEDetectionHarmonizer


class RSNAPEDetectionTrainHarmonizer(RSNAPEDetectionHarmonizer):
    """Train-split subclass of :class:`RSNAPEDetectionHarmonizer`.

    It has the same logic as the base class. It is a separate class so that a saved
    harmonizer loads back into the train class.
    """
