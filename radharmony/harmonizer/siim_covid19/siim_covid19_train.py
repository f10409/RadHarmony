"""Train-split harmonizer for the SIIM COVID-19 Detection Challenge."""

from .siim_covid19 import SIIMCOVID19Harmonizer


class SIIMCOVID19TrainHarmonizer(SIIMCOVID19Harmonizer):
    """Train-split harmonizer; identical to base for save/load round-trip identity."""
    pass
