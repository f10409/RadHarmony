"""Train-split harmonizer for the RSNA Pneumonia Detection Challenge (Kaggle Stage 2)."""

from .._split import _LoadOldFileMixin
from .rsna_pneumonia_kaggle import RSNAPneumoniaKaggleHarmonizer


class RSNAPneumoniaKaggleTrainHarmonizer(_LoadOldFileMixin, RSNAPneumoniaKaggleHarmonizer):
    """Train split. It has the same logic as the base class.

    ``load_from_saved()`` also loads a file that :class:`RSNAPneumoniaKaggleHarmonizer`
    saved. That class reads only the train files.
    """

    OLD_FILE_CLASS = RSNAPneumoniaKaggleHarmonizer
