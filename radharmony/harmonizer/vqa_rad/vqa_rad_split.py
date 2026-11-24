"""Harmonizers for the official VQA-RAD train and test questions."""

from .._split import _KeepSplitMixin
from .vqa_rad import VQARadHarmonizer


class VQARadTrainHarmonizer(_KeepSplitMixin, VQARadHarmonizer):
    """VQA-RAD train questions (``phrase_type`` is ``freeform`` or ``para``)."""

    SPLIT = "train"


class VQARadTestHarmonizer(_KeepSplitMixin, VQARadHarmonizer):
    """VQA-RAD test questions (``phrase_type`` is ``test_freeform`` or ``test_para``, 451 questions).

    202 of the 203 test images also have train questions. The paper made
    the test set this way, so the split is by question, not by image.
    """

    SPLIT = "test"
