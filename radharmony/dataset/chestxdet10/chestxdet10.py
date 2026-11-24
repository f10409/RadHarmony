"""MONAI datasets for ChestX-Det10: instance-level box annotations on CXR.

See :class:`~radharmony.harmonizer.ChestXDet10Harmonizer` for the four traps in
the released data:

* The ``train-old/`` directory name.
* Filenames that do NOT match NIH ChestX-ray14.
* 40 RGBA images among 3,503 grayscale images.
* 764 negative images that are real rows, not missing data.
"""
import torch

from radharmony.harmonizer import ChestXDet10Harmonizer, ChestXDet10TestHarmonizer, ChestXDet10TrainHarmonizer
from radharmony.harmonizer._split import _FixedSplitArgMixin
from radharmony.harmonizer.chestxdet10 import CLASSES
from radharmony.registry import register_dataset

from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("chestxdet10")
class ChestXDet10Dataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for ChestX-Det10.

    Args:
        base_image_dir: Dataset root holding ``train-old/`` and ``test_data/``.
        split: ``train`` (3,001 images) or ``test`` (542).
        csv_path: Annotation JSON. Defaults to ``<root>/<split>.json``.
        keep_negatives: Keep the images with no findings (default True), 681 in
            train, 83 in test. They are the benchmark's negative controls.
        output_cls: Yield the 10-way multi-label vector under ``cls``.
        output_bbox: Yield boxes under ``bbox`` as normalized
            [dim0_min, dim0_max, dim1_min, dim1_max].
        output_mask / output_report / output_reg: Not supported. A warning, and no key (D-11).

    The official metric is recall at fixed false positives per image
    (0.05 / 0.1 / 0.2 FP/image at IoU 0.5), not mAP. See ``eval.py`` in the
    authors' repo. Nothing here computes it. This class only serves data.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = [c.lower() for c in CLASSES]
    _HARMONIZER_CLS = ChestXDet10Harmonizer
    _SPLIT = "train"

    def __init__(self, base_image_dir: str = None, split: str = None,
                 csv_path: str = None, transform=None, cache_dir: str = "./cache",
                 output_cls: bool = False, output_bbox: bool = False,
                 keep_negatives: bool = True, harmonized_df=None, harmonizer=None,
                 harmonizer_path: str = None, dtype=torch.bfloat16,
                 output_mask: bool = False, output_report: bool = False, output_reg: bool = False):
        # A saved harmonizer knows its image folder.
        if harmonizer_path is not None and base_image_dir is None:
            base_image_dir = self._HARMONIZER_CLS.load_from_saved(harmonizer_path).base_image_dir
        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "Give base_image_dir, or give harmonizer_path, harmonizer or harmonized_df."
            )
        split = split or self._SPLIT
        if base_image_dir is not None:
            csv_path = csv_path or f"{base_image_dir.rstrip('/')}/{split}.json"
        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")
        if output_bbox:
            output_keys.add("bbox")
            output_keys.add("bbox_labels")
        if transform is None:
            # 40 of the 3,543 released images are RGBA gray images (R == G == B, alpha 255).
            # The channel step of the base load makes them 1 channel (D-42).
            transform = RadiologyTransform2D(output_keys=output_keys, dtype=dtype).get_transform()

        super().__init__(base_image_dir=base_image_dir, transform=transform,
                         cache_dir=cache_dir, output_cls=output_cls,
                         output_bbox=output_bbox,
                         # not in SUPPORTED_OUTPUTS: the base class warns
                         output_mask=output_mask, output_report=output_report, output_reg=output_reg,
                         harmonized_df=harmonized_df, harmonizer=harmonizer,
                         harmonizer_path=harmonizer_path)
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = self._HARMONIZER_CLS(
                csv_path=csv_path, base_image_dir=base_image_dir,
                split=split, keep_negatives=keep_negatives)

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()


@register_dataset("chestxdet10_train")
class ChestXDet10TrainDataset(_FixedSplitArgMixin, ChestXDet10Dataset):
    """ChestX-Det10 training split: 3,001 images, 6,863 boxes, 681 negatives.

    ``split="test"`` raises ``ValueError``: this class gives only the train split.
    """

    SPLIT = "train"
    _HARMONIZER_CLS = ChestXDet10TrainHarmonizer


@register_dataset("chestxdet10_test")
class ChestXDet10TestDataset(_FixedSplitArgMixin, ChestXDet10Dataset):
    """ChestX-Det10 test split: 542 images, 1,476 boxes, 83 negatives.

    ``split="train"`` raises ``ValueError``: this class gives only the test split.
    """

    SPLIT = "test"
    _HARMONIZER_CLS = ChestXDet10TestHarmonizer
