"""Train-split harmonizer for the SIIM-ACR Pneumothorax dataset."""

from .siim_acr_ptx import SIIMACRPTXHarmonizer


class SIIMACRPTXTrainHarmonizer(SIIMACRPTXHarmonizer):
    """Train-split harmonizer; identical to base for save/load round-trip identity."""
    pass
