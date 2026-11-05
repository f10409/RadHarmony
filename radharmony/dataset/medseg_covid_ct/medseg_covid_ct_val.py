"""MONAI dataset for the 10 MedSeg dataset1 ``val_im`` slices (no masks, D-25)."""

import os
import warnings

import pandas as pd
import torch

from radharmony.harmonizer.medseg_covid_ct import MedSegCOVIDCTValHarmonizer
from radharmony.harmonizer.medseg_covid_ct.medseg_covid_ct import slice_folder
from radharmony.registry import register_dataset
from .medseg_covid_ct import MedSegCOVIDCTDataset


@register_dataset("medseg_covid_ct_val")
class MedSegCOVIDCTValDataset(MedSegCOVIDCTDataset):
    """MedSeg Covid Dataset 1 ``val_im`` slices: images only, no labels and no masks.

    Every ``output_*`` flag gives a warning and no key.

    Args:
        base_image_dir: The folder that holds ``val_im.nii.gz``.
        slice_output_dir: The folder for the slice PNGs.
        The other arguments are the same as :class:`MedSegCOVIDCTDataset`.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset()
    LABEL_COLS = []
    _HARMONIZER_CLS = MedSegCOVIDCTValHarmonizer

    def __init__(self, base_image_dir: str = None, slice_output_dir: str = None,
                 hu_window=(-1000.0, 400.0), transform=None, cache_dir: str = "./cache",
                 output_cls: bool = False, output_mask: bool = False, harmonized_df=None,
                 harmonizer=None, harmonizer_path: str = None, dtype=torch.bfloat16,
                 output_report: bool = False, output_reg: bool = False, output_bbox: bool = False):
        if output_cls or output_mask or output_report or output_bbox or output_reg:
            warnings.warn("The MedSeg val_im slices have no labels, masks, reports or boxes. "
                          "The output_* flags are ignored.", UserWarning, stacklevel=2)
        super().__init__(base_image_dir=base_image_dir, slice_output_dir=slice_output_dir,
                         hu_window=hu_window, transform=transform, cache_dir=cache_dir,
                         harmonized_df=harmonized_df, harmonizer=harmonizer,
                         harmonizer_path=harmonizer_path, dtype=dtype)
        # The parent read the slice folder and the window, also from a harmonizer (V-H1-42).
        if self._slice_output_dir:
            self.base_image_dir = os.path.join(os.path.expanduser(self._slice_output_dir),
                                               slice_folder(MedSegCOVIDCTValHarmonizer.SLICE_FOLDER, self._hu_window),
                                               "images")

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._HARMONIZER_CLS(base_dir=self._src_dir, slice_output_dir=self._slice_output_dir,
                                    hu_window=self._hu_window).harmonize()
