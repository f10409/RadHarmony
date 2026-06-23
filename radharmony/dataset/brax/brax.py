"""MONAI datasets for the BRAX (Brazilian Chest X-Ray) v1.1.0 release.

BRAX ships both DICOM (under ``Anonymized_DICOMs/``) and PNG (under
``images/``) variants in the same PhysioNet download.  Two thin dataset
classes share a single :class:`BRAXHarmonizer`:

* :class:`BRAXDataset` — DICOM, ``image_format="dicom"``
* :class:`BRAXPNGDataset` — PNG, ``image_format="png"``

Both expect ``base_image_dir`` to point at the **BRAX root** (the directory
containing ``master_spreadsheet.csv``, ``Anonymized_DICOMs/``, and
``images/``).  ``image_path`` in the harmonized DataFrame includes the
``Anonymized_DICOMs/`` or ``images/`` prefix exactly as written in the
master CSV — joining ``base_image_dir`` + ``image_path`` resolves to the
on-disk file.

Labels are CheXpert-aligned (14 classes).  ``uncertain_strategy`` controls
how the labeller's ``-1`` (uncertain) cells are mapped — the default
``"u_zeros"`` retains every row by treating uncertain mentions as
negative (matches the most-cited CheXpert-paper baseline).  Reports are
not distributed in BRAX 1.1.0; ``output_report`` is ignored.
"""

import os
import warnings

import pandas as pd
import torch

from radharmony.harmonizer import BRAXHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


_LABEL_COLS = [
    "atelectasis",
    "cardiomegaly",
    "consolidation",
    "edema",
    "enlarged_cardiomediastinum",
    "fracture",
    "lung_lesion",
    "lung_opacity",
    "no_finding",
    "pleural_effusion",
    "pleural_other",
    "pneumonia",
    "pneumothorax",
    "support_devices",
]


class _BRAXDatasetBase(BaseRadiologicalDataset):
    """Shared logic for the two BRAX dataset variants.

    Args:
        base_image_dir: BRAX root (e.g.
            ``/data/physionet.org/files/brax/1.1.0/``).  Contains
            ``master_spreadsheet.csv``, ``Anonymized_DICOMs/``, and ``images/``.
        csv_path: Path to ``master_spreadsheet.csv``.  Auto-inferred near
            ``base_image_dir`` when ``None``.
        uncertain_strategy: How to map ``-1`` (uncertain) label cells.
            One of ``"raw"`` (default — preserves the source ``1 / 0 / -1
            / NaN`` encoding exactly), ``"u_zeros"``, ``"u_ones"``,
            ``"u_ignore"``, ``"drop"``.  See :class:`BRAXHarmonizer` for
            semantics.  Note: with ``"raw"`` and ``output_cls=True`` the
            yielded ``cls`` tensor will contain ``-1`` for uncertain
            findings; downstream loss functions must handle it.
        transform: MONAI Compose transform.  Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask / output_report / output_bbox: Not supported; warn-and-ignore.
    """

    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = BRAXHarmonizer
    _IMAGE_FORMAT: str = "dicom"  # overridden in subclasses

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        uncertain_strategy: str = "raw",
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if csv_path:
            csv_path = os.path.expanduser(csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = BRAXHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_image_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided. "
                "Pass the BRAX root (the directory containing master_spreadsheet.csv)."
            )

        if output_mask:
            warnings.warn(
                "BRAX does not provide segmentation masks. output_mask will be ignored.",
                UserWarning, stacklevel=2,
            )
            output_mask = False
        if output_report:
            warnings.warn(
                "BRAX 1.1.0 does not distribute the source radiology reports. "
                "output_report will be ignored.",
                UserWarning, stacklevel=2,
            )
            output_report = False
        if output_bbox:
            warnings.warn(
                "BRAX does not provide bounding boxes. output_bbox will be ignored.",
                UserWarning, stacklevel=2,
            )
            output_bbox = False

        self._uncertain_strategy = uncertain_strategy

        if output_cls and uncertain_strategy == "raw":
            warnings.warn(
                "BRAX uncertain_strategy='raw' preserves -1 values; the 'cls' "
                "tensor will contain -1 for uncertain findings. If your loss "
                "doesn't handle -1, pass uncertain_strategy='u_zeros' / "
                "'u_ones' / 'u_ignore' / 'drop' to coerce it.",
                UserWarning, stacklevel=2,
            )

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = (
                infer_path(
                    base_image_dir,
                    "master_spreadsheet.csv",
                    "master_spreadsheet_update.csv",
                    user_path=csv_path or "",
                )
                or csv_path
            )
            if not self._csv_path:
                raise FileNotFoundError(
                    f"Could not locate master_spreadsheet.csv near {base_image_dir!r}; "
                    "pass csv_path= explicitly."
                )
        else:
            self._csv_path = csv_path

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
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
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        h = BRAXHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            image_format=self._IMAGE_FORMAT,
            uncertain_strategy=self._uncertain_strategy,
        )
        return h.harmonize()


@register_dataset("brax")
class BRAXDataset(_BRAXDatasetBase):
    """BRAX with the DICOM image variant.

    ``base_image_dir`` should point at the BRAX root (containing
    ``Anonymized_DICOMs/`` and ``master_spreadsheet.csv``).  Images are
    16-bit DICOMs read via MONAI's ITKReader.
    """

    _IMAGE_FORMAT = "dicom"


@register_dataset("brax_png")
class BRAXPNGDataset(_BRAXDatasetBase):
    """BRAX with the PNG image variant.

    ``base_image_dir`` should point at the BRAX root (containing
    ``images/`` and ``master_spreadsheet.csv``).  Images are 16-bit PNGs.
    Loads faster than the DICOM variant for large training runs.
    """

    _IMAGE_FORMAT = "png"
