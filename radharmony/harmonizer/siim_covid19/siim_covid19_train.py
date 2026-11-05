"""Train-split harmonizer for the SIIM COVID-19 Detection Challenge."""

from .._split import _LoadOldFileMixin
from .siim_covid19 import SIIMCOVID19Harmonizer


class SIIMCOVID19TrainHarmonizer(_LoadOldFileMixin, SIIMCOVID19Harmonizer):
    """Train split. It has the same logic as the base class.

    ``load_from_saved()`` also loads a file that :class:`SIIMCOVID19Harmonizer` saved.
    That class reads only the train files.
    """

    OLD_FILE_CLASS = SIIMCOVID19Harmonizer
