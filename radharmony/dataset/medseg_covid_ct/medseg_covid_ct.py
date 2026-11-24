"""MONAI dataset for the MedSeg COVID-19 CT segmentation sets (2-D slices)."""

import os

import pandas as pd
import torch

from radharmony.harmonizer.medseg_covid_ct import MedSegCOVIDCTHarmonizer
from radharmony.harmonizer.medseg_covid_ct.medseg_covid_ct import slice_folder
from radharmony.registry import register_dataset
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D

_LABEL_COLS = ["any_lesion", "consolidation", "ground_glass", "pleural_effusion"]


@register_dataset("medseg_covid_ct")
class MedSegCOVIDCTDataset(BaseRadiologicalDataset):
    """2-D slice dataset over the MedSeg COVID-19 CT releases.

    The CT counterpart to QaTa-COV19, and the source behind the "MosMedData+"
    benchmark used by the LViT / DD-CMD text-guided segmentation line.

    Two subsets, selected with ``subset``:

    - ``"dataset1"``: 100 annotated axial slices (``tr_im``/``tr_mask``),
      512x512. ``Test-Images-Clinical-Details.csv`` maps them to 43 SIRM cases.
      ``patient_id`` is the SIRM case number, and the table has ``age_years``
      and ``sex`` (D-10, D-06, D-26).
    - ``"dataset2"``: 9 whole volumes (``rp_im``/``rp_msk``). Here
      ``patient_id`` is the volume index and IS a real grouping. Split on it,
      because slices within a volume are strongly correlated.

    Unlike QaTa-COV19, the labels here are not constant. The released masks are
    class-index volumes (1 = ground-glass, 2 = consolidation, 3 = pleural
    effusion, dataset1 only). So each slice gets real multi-label presence
    flags plus ``any_lesion``, which is 0 on slices with no annotated lesion.

    CT volumes cannot be read by the 2-D pipeline, so the harmonizer writes one
    PNG per slice into ``slice_output_dir`` on first use (idempotent). Images
    are rendered through a lung window (default ``[-1000, 400]`` HU). The
    ``mask`` output is the binary any-lesion union.

    Args:
        base_image_dir: Folder holding the release, containing
            ``tr_im.nii.gz`` for ``dataset1``, or ``rp_im/`` for ``dataset2``.
        slice_output_dir: Where per-slice PNGs are written. Required unless a
            pre-harmonized df / saved harmonizer is supplied.
        subset: ``"dataset1"`` (default) or ``"dataset2"``.
        hu_window: ``(low, high)`` Hounsfield window for the PNGs. It is part of the slice folder name.
        transform: MONAI Compose. Defaults to the standard 2-D 224 px pipeline.
        cache_dir: PersistentDataset cache dir; ``None`` disables.
        output_cls: Yield the 4 label flags under ``"cls"``.
        output_mask: Yield the binary lesion mask under ``"mask"``.
        output_report: Not supported: no reports in this release.
        output_bbox: Not supported.
        dtype: Dtype of ``img``. The labels stay float32. Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "mask"})
    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = MedSegCOVIDCTHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        slice_output_dir: str = None,
        subset: str = "dataset1",
        hu_window=(-1000.0, 400.0),
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
        preset = harmonizer_path is not None or harmonized_df is not None or harmonizer is not None

        src = self._HARMONIZER_CLS.load_from_saved(harmonizer_path) if harmonizer_path is not None else harmonizer
        if src is not None:
            # The table of a harmonizer belongs to its own subset, window and slice folder (V-H1-42).
            base_image_dir = base_image_dir or getattr(src, "base_dir", None)
            slice_output_dir = slice_output_dir or getattr(src, "slice_output_dir", None)
            subset = getattr(src, "subset", subset)
            hu_window = tuple(getattr(src, "hu_window", hu_window))

        if base_image_dir is None and not preset:
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the folder containing tr_im.nii.gz (subset='dataset1') or "
                "rp_im/ (subset='dataset2')."
            )
        if slice_output_dir is None and not preset:
            raise ValueError(
                "slice_output_dir is required. MedSeg ships CT volumes (.nii.gz); "
                "one PNG per slice is written there once before MONAI can load "
                "them. Pass slice_output_dir='/path/to/medseg_slices'."
            )

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_mask:
                output_keys.add("mask")
            transform = RadiologyTransform2D(
                img_size=224, output_keys=output_keys, dtype=dtype,
            ).get_transform()

        # image_path is a bare filename under <slice_output_dir>/<subset>_hu<low>_<high>/images/
        resolved_base = (
            os.path.join(os.path.expanduser(slice_output_dir), slice_folder(subset, hu_window), "images")
            if slice_output_dir else base_image_dir
        )

        super().__init__(
            base_image_dir=resolved_base,
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
        self._src_dir = base_image_dir
        self._slice_output_dir = slice_output_dir
        self._subset = subset
        self._hu_window = hu_window

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        h = MedSegCOVIDCTHarmonizer(
            base_dir=self._src_dir,
            slice_output_dir=self._slice_output_dir,
            subset=self._subset,
            hu_window=self._hu_window,
        )
        return h.harmonize()
