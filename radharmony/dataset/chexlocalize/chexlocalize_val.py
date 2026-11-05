"""MONAI dataset for the CheXlocalize validation split."""

from radharmony.harmonizer import CheXlocalizeValHarmonizer
from radharmony.registry import register_dataset
from .chexlocalize import CheXlocalizeDataset


@register_dataset("chexlocalize_val")
class CheXlocalizeValDataset(CheXlocalizeDataset):
    """CheXlocalize validation split (234 images of the CheXpert valid set).

    ``base_image_dir`` is the ``val/`` folder. The class looks for
    ``val_labels.csv`` and ``gt_segmentations_val.json``. The other arguments
    are the same as :class:`CheXlocalizeDataset` (the test split).
    """

    _HARMONIZER_CLS = CheXlocalizeValHarmonizer
    _LABEL_CSV = "val_labels.csv"
    _MASK_JSON = "gt_segmentations_val.json"
