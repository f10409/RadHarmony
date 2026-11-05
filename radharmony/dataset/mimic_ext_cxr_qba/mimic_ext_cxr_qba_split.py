"""VQA datasets for the official MIMIC-Ext-CXR-QBA train, validate and test splits (D-09)."""

from radharmony.harmonizer import (
    MIMICExtCXRQBATestHarmonizer,
    MIMICExtCXRQBATrainHarmonizer,
    MIMICExtCXRQBAValidateHarmonizer,
)
from radharmony.registry import register_dataset
from .mimic_ext_cxr_qba import MIMICExtCXRQBADataset


@register_dataset("mimic_ext_cxr_qba_train")
class MIMICExtCXRQBATrainDataset(MIMICExtCXRQBADataset):
    """MIMIC-Ext-CXR-QBA train patients. The arguments are the same as :class:`MIMICExtCXRQBADataset`."""

    _HARMONIZER_CLS = MIMICExtCXRQBATrainHarmonizer


@register_dataset("mimic_ext_cxr_qba_validate")
class MIMICExtCXRQBAValidateDataset(MIMICExtCXRQBADataset):
    """MIMIC-Ext-CXR-QBA validate patients. The arguments are the same as :class:`MIMICExtCXRQBADataset`."""

    _HARMONIZER_CLS = MIMICExtCXRQBAValidateHarmonizer


@register_dataset("mimic_ext_cxr_qba_test")
class MIMICExtCXRQBATestDataset(MIMICExtCXRQBADataset):
    """MIMIC-Ext-CXR-QBA test patients. The arguments are the same as :class:`MIMICExtCXRQBADataset`."""

    _HARMONIZER_CLS = MIMICExtCXRQBATestHarmonizer
