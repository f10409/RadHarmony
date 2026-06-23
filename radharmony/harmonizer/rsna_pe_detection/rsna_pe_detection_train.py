"""Train-split harmonizer for the RSNA PE Detection Challenge (2020)."""

from .rsna_pe_detection import RSNAPEDetectionHarmonizer


class RSNAPEDetectionTrainHarmonizer(RSNAPEDetectionHarmonizer):
    """Train-split subclass of :class:`RSNAPEDetectionHarmonizer`.

    Identical harmonization logic; distinct class for save/load round-trip
    identity.
    """
