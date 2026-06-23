"""Train-split harmonizer for the RSNA Pneumonia Detection Challenge (Kaggle Stage 2)."""

from .rsna_pneumonia_kaggle import RSNAPneumoniaKaggleHarmonizer


class RSNAPneumoniaKaggleTrainHarmonizer(RSNAPneumoniaKaggleHarmonizer):
    """Train-split harmonizer; identical to base for save/load round-trip identity."""
    pass
