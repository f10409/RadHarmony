"""Train-split harmonizer for the SIIM-ACR Pneumothorax dataset."""

from .._split import _LoadOldFileMixin
from .siim_acr_ptx import SIIMACRPTXHarmonizer


class SIIMACRPTXTrainHarmonizer(_LoadOldFileMixin, SIIMACRPTXHarmonizer):
    """Train split. It has the same logic as the base class.

    ``load_from_saved()`` also loads a file that :class:`SIIMACRPTXHarmonizer` saved.
    That class reads only the train files.
    """

    OLD_FILE_CLASS = SIIMACRPTXHarmonizer
