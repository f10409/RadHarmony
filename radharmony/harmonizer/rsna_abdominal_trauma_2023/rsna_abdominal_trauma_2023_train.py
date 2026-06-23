"""Train-split harmonizer for the RSNA 2023 Abdominal Trauma Detection Challenge."""

from .rsna_abdominal_trauma_2023 import RSNAAbdominalTrauma2023Harmonizer


class RSNAAbdominalTrauma2023TrainHarmonizer(RSNAAbdominalTrauma2023Harmonizer):
    """Train-split subclass of :class:`RSNAAbdominalTrauma2023Harmonizer`.

    Identical harmonization logic; distinct class for save/load round-trip
    identity.
    """
