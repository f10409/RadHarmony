"""Harmonizer for the NIH ChestX-ray14 test split."""

from .chestxray14 import ChestXray14Harmonizer


class ChestXray14TestHarmonizer(ChestXray14Harmonizer):
    """ChestX-ray14 test split (25,596 images).

    Keeps the images listed in ``test_list.txt``, which NIH ships with the
    dataset. This includes the 880 images with boxes, but the class gives no
    boxes (use :class:`ChestXray14BboxHarmonizer`). If the file is not found,
    or ``base_image_dir`` is None, ``harmonize()`` raises an error.

    Both train and test splits carry full labels from ``Data_Entry_2017.csv``.

    Args:
        csv_path: Path to ``Data_Entry_2017.csv``.
        bbox_csv_path: Not used. The official list decides the rows.
        base_image_dir: Root directory containing ``images_*/images/`` folders.
    """

    SPLIT_LIST = "test_list.txt"
