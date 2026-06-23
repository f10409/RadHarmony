"""Train-split harmonizer for the RSNA 2024 Lumbar Spine Degenerative Classification."""

from .rsna_2024_lumbar_spine import RSNA2024LumbarSpineHarmonizer


class RSNA2024LumbarSpineTrainHarmonizer(RSNA2024LumbarSpineHarmonizer):
    """Train-split subclass of :class:`RSNA2024LumbarSpineHarmonizer`.

    Identical harmonization logic; distinct class for save/load round-trip
    identity (``load_from_saved`` can reconstruct the correct subclass).
    """
