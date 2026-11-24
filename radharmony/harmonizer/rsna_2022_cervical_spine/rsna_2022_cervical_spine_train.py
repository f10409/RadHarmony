"""Train-split harmonizer for the RSNA 2022 Cervical Spine Fracture Detection Challenge."""

from .rsna_2022_cervical_spine import RSNA2022CervicalSpineHarmonizer


class RSNA2022CervicalSpineTrainHarmonizer(RSNA2022CervicalSpineHarmonizer):
    """Train-split subclass of :class:`RSNA2022CervicalSpineHarmonizer`.

    It has the same logic as the base class. It is a separate class so that a saved
    harmonizer loads back into the train class.
    """
