"""Harmonizer for the NIH ChestX-ray14 training split."""

from .chestxray14 import ChestXray14Harmonizer


class ChestXray14TrainHarmonizer(ChestXray14Harmonizer):
    """ChestX-ray14 training split (86,524 images).

    Keeps the images listed in ``train_val_list.txt``, which NIH ships with
    the dataset. If the file is not found, or ``base_image_dir`` is None,
    ``harmonize()`` raises an error.

    Args:
        csv_path: Path to ``Data_Entry_2017.csv``.
        bbox_csv_path: Not used. The official list decides the rows.
        base_image_dir: Root directory containing ``images_*/images/`` folders.
    """

    SPLIT_LIST = "train_val_list.txt"
