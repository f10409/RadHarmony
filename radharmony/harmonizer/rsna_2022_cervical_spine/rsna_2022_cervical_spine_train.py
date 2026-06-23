"""Train-split harmonizer for the RSNA 2022 Cervical Spine Fracture Detection Challenge."""

from .rsna_2022_cervical_spine import RSNA2022CervicalSpineHarmonizer


class RSNA2022CervicalSpineTrainHarmonizer(RSNA2022CervicalSpineHarmonizer):
    """Train-split subclass of :class:`RSNA2022CervicalSpineHarmonizer`.

    Identical harmonization logic; distinct class for save/load round-trip
    identity.
    """
