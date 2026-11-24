"""MONAI datasets for NODE21: nodule boxes on frontal chest radiographs.

See :class:`~radharmony.harmonizer.Node21Harmonizer` for the six traps in the
released data:

* The misspelled ``proccessed_data`` directory.
* Images nested in ``images/``.
* Negatives encoded as a zero-filled box row.
* One row per box, not per image.
* A ragged ``original_data``.
* Box coordinates that are only valid against their own variant.

Images are ``.mha``, which MONAI reads through ITK, unlike the PNG/DICOM
datasets elsewhere in the repo. Pixels are uint16 (12-bit, observed 0..4095),
so the percentile windowing in :class:`RadiologyTransform2D` is doing real work
here: a plain 8-bit cast would clip almost everything to black.
"""
import torch

from radharmony.harmonizer import Node21Harmonizer
from radharmony.harmonizer.node21 import CLASSES
from radharmony.registry import register_dataset

from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("node21")
class Node21Dataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for NODE21.

    Args:
        base_image_dir: Dataset root containing ``cxr_images/``.
        variant: ``processed`` (default, uniform 1024x1024 uint16) or
            ``original``, which is ragged in both shape and dtype.
        csv_path: ``metadata.csv``. Defaults to the one inside the variant dir.
        keep_negatives: Keep the 3,748 nodule-free controls (default True).
            They are the benchmark's negative class, not missing data.
        output_cls: Yield the single-element nodule-presence vector under
            ``cls``.
        output_bbox: Yield boxes under ``bbox`` as normalized
            [dim0_min, dim0_max, dim1_min, dim1_max].
        output_mask / output_report / output_reg: Not supported. A warning, and no key (D-11).

    The public release is the challenge's *training* data. The evaluation set is
    held out on Grand Challenge and is not distributed. There is therefore no
    train/test split here, and a published "NODE21 test" number is not
    reproducible from this data alone.

    The official metric is a free-response ROC (sensitivity against average
    false positives per image), not mAP. Nothing here computes it. This class
    only serves data.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox"})
    LABEL_COLS = CLASSES
    _HARMONIZER_CLS = Node21Harmonizer
    _VARIANT = "processed"

    def __init__(self, base_image_dir: str = None, variant: str = None,
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
        variant = variant or self._VARIANT
        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")
        if output_bbox:
            output_keys.add("bbox")
            output_keys.add("bbox_labels")
        if transform is None:
            # .mha decodes to a single-channel 2-D array, so the standard 2-D
            # pipeline applies unchanged, no channel slice is needed here,
            # unlike ChestX-Det10's RGBA images.
            transform = RadiologyTransform2D(output_keys=output_keys,
                                             dtype=dtype).get_transform()

        super().__init__(base_image_dir=base_image_dir, transform=transform,
                         cache_dir=cache_dir, output_cls=output_cls,
                         output_bbox=output_bbox,
                         # not in SUPPORTED_OUTPUTS: the base class warns
                         output_mask=output_mask, output_report=output_report, output_reg=output_reg,
                         harmonized_df=harmonized_df, harmonizer=harmonizer,
                         harmonizer_path=harmonizer_path)
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = Node21Harmonizer(
                csv_path=csv_path, base_image_dir=base_image_dir,
                variant=variant, keep_negatives=keep_negatives)

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()


@register_dataset("node21_original")
class Node21OriginalDataset(Node21Dataset):
    """NODE21 served from ``original_data`` rather than the preprocessed copy.

    Sizes and dtypes are mixed here (sampled: 1024x1024, 2828x2320, 2991x3000,
    uint8 and uint16). The box coordinates differ from the processed variant's.
    They are only valid against these pixels.
    """

    _VARIANT = "original"
