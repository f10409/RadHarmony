"""Harmonizers for the official VinDr-PCXR train and test splits."""

from .vindr_pcxr import VinDrPCXRHarmonizer


class VinDrPCXRTrainHarmonizer(VinDrPCXRHarmonizer):
    """VinDr-PCXR official train split (``image_labels_train.csv`` and ``train/``)."""

    SPLITS = ("train",)


class VinDrPCXRTestHarmonizer(VinDrPCXRHarmonizer):
    """VinDr-PCXR official test split (``image_labels_test.csv`` and ``test/``)."""

    SPLITS = ("test",)
