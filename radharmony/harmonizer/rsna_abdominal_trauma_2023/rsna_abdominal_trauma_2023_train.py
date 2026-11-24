"""Train-split harmonizer for the RSNA 2023 Abdominal Trauma Detection Challenge."""

from .rsna_abdominal_trauma_2023 import RSNAAbdominalTrauma2023Harmonizer


class RSNAAbdominalTrauma2023TrainHarmonizer(RSNAAbdominalTrauma2023Harmonizer):
    """Train-split subclass of :class:`RSNAAbdominalTrauma2023Harmonizer`.

    It has the same logic as the base class. It is a separate class so that a saved
    harmonizer loads back into the train class.
    """
