"""MONAI datasets for the official QaTa-COV19 v2 train and test splits."""

from radharmony.harmonizer import QaTaCOV19TestHarmonizer, QaTaCOV19TrainHarmonizer
from radharmony.registry import register_dataset
from .qata_cov19 import QaTaCOV19Dataset


@register_dataset("qata_cov19_train")
class QaTaCOV19TrainDataset(QaTaCOV19Dataset):
    """QaTa-COV19 v2 train split. ``base_image_dir`` must be ``Train Set/Images/``, else ``ValueError``."""

    _HARMONIZER_CLS = QaTaCOV19TrainHarmonizer


@register_dataset("qata_cov19_test")
class QaTaCOV19TestDataset(QaTaCOV19Dataset):
    """QaTa-COV19 v2 test split. ``base_image_dir`` must be ``Test Set/Images/``, else ``ValueError``."""

    _HARMONIZER_CLS = QaTaCOV19TestHarmonizer
