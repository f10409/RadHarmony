"""MONAI dataset for QaTa-COV19 (v2): COVID-19 CXRs + infection-region masks."""

import os
import warnings

import pandas as pd
import torch

from radharmony.harmonizer.qata_cov19 import QaTaCOV19Harmonizer
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


_LABEL_COLS = ["covid"]


@register_dataset("qata_cov19")
class QaTaCOV19Dataset(BaseRadiologicalDataset):
    """PyTorch/MONAI dataset for one split of QaTa-COV19-v2.

    9,258 COVID-19 chest X-rays (224x224 8-bit grayscale PNG). The publishers
    split them into 7,145 train / 2,113 test. Each image has one binary
    ground-truth mask of the COVID-19-infected lung region.

    **One class serves both splits**, they differ only in which directory you
    pass as ``base_image_dir``, so there is nothing for split-specific
    subclasses to override:

    - train: ``.../QaTa-COV19/QaTa-COV19-v2/Train Set/Images/``
    - test:  ``.../QaTa-COV19/QaTa-COV19-v2/Test Set/Images/``

    The subject ids of the two splits do not overlap (1,656 train, 474 test).
    But 1 test picture is also a train picture, and 37 pairs of train files are
    near copies (not pixel-equal) with 2 ``patient_id`` values. So
    ``get_datasets()`` can put the 2 copies in train and val (D-10, D-34). The
    wiki page gives the file names.

    Masks are already binary PNGs in the split's sibling ``Ground-truths/``
    directory. Unlike SIIM-ACR PTX, Montgomery or Shenzhen, there is no
    ``mask_output_dir`` to pass and no decoding step. ``output_mask=True`` just
    works.

    ``cls`` is a single ``covid`` column that is **constant 1**, every image
    in v2 is COVID-positive. It is there so ``output_cls`` yields something and
    so this dataset composes with a negative-control set. It is not a
    classification task on its own.

    Args:
        base_image_dir: A split's ``Images/`` directory (see above). Required
            unless ``harmonizer_path`` or ``harmonized_df`` is provided.
        masks_dir: The split's ``Ground-truths/`` directory. Defaults to the
            sibling ``../Ground-truths/`` of ``base_image_dir``, which is
            correct for an unmodified download.
        transform: MONAI Compose transform. Defaults to the standard 2-D
            224 px pipeline.
        cache_dir: MONAI PersistentDataset cache directory. ``None`` disables
            caching.
        output_cls: Yield the ``covid`` label vector under the ``"cls"`` key.
        output_mask: Yield the infection-region mask under the ``"mask"`` key.
        output_report: Not supported: QaTa-COV19 ships no reports.
        output_bbox: Not supported: QaTa-COV19 ships no bounding boxes.
        dtype: Dtype of ``img``. The labels stay float32. Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask"})
    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = QaTaCOV19Harmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        masks_dir: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)

        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_dir", None)
            if masks_dir is None:
                masks_dir = getattr(_h, "masks_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the path to a QaTa-COV19-v2 split's Images/ directory, e.g. "
                "'.../QaTa-COV19/QaTa-COV19-v2/Train Set/Images/'."
            )

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_mask:
                output_keys.add("mask")
            transform = RadiologyTransform2D(
                img_size=224,
                output_keys=output_keys,
                dtype=dtype,
            ).get_transform()

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )
        self._masks_dir = masks_dir

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        h = self._HARMONIZER_CLS(
            base_dir=self.base_image_dir,
            masks_dir=self._masks_dir,
        )
        df = h.harmonize()
        # Every v2 image is masked in an unmodified download. A gap means the
        # Ground-truths/ dir is missing or was pruned, and output_mask=True
        # would silently drop those rows in get_data_dict.  Say so once here
        # rather than let the row count quietly shrink.
        if self.output_mask and "mask_path" in df.columns:
            n_missing = int(df["mask_path"].isna().sum())
            if n_missing:
                warnings.warn(
                    f"{n_missing} of {len(df)} QaTa-COV19 images have no mask in "
                    f"{self._masks_dir or 'the sibling Ground-truths/ directory'}; "
                    "those rows will be dropped when output_mask=True. An "
                    "unmodified QaTa-COV19-v2 download pairs every image 1:1 "
                    "with a mask.",
                    UserWarning,
                    stacklevel=2,
                )
        return df
