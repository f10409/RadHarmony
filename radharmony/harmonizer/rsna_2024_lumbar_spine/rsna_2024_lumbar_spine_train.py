"""Train-split harmonizer for the RSNA 2024 Lumbar Spine Degenerative Classification."""

from .rsna_2024_lumbar_spine import RSNA2024LumbarSpineHarmonizer


class RSNA2024LumbarSpineTrainHarmonizer(RSNA2024LumbarSpineHarmonizer):
    """Train-split subclass of :class:`RSNA2024LumbarSpineHarmonizer`.

    It has the same logic as the base class. It is a separate class so that a saved
    harmonizer loads back into the train class.
    """
