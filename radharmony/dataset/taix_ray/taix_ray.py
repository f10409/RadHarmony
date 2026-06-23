"""MONAI datasets for the TAIX-Ray bedside chest radiography dataset.

TAIX-Ray ships in two configurations on HuggingFace:

* ``data/`` — 512-pixel resized PNGs (~58 GB, 215,381 images)
* ``original/`` — native-resolution PNGs (~1.27 TB, same images)

Both share the same metadata schema, harmonizer, and label set; only the
on-disk image directory differs.  Use :class:`TAIXRay512Dataset` for the
former and :class:`TAIXRayDataset` for the latter.

Both classes expose two label representations via the ``label_mode`` arg:

* ``"binary"`` (default) — any severity ≥ 1 maps to 1.0; matches the
  multi-label-binary convention used by CheXpert / MIMIC datasets and
  plugs into a standard BCE training loop.
* ``"ordinal"`` — preserves the raw 0–4 severity grades (0–3 for heart
  size); intended for ordinal-regression heads (e.g. CORN loss as in
  Truhn et al. 2026).

Both classes also ship a :meth:`get_datasets_predefined` method that
honours the dataset's own ``split`` column (the paper's official
patient-level train / val / test partition).
"""

import os
from pathlib import Path

import pandas as pd
import torch

from radharmony.harmonizer import TAIXRayHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.data_utils import get_data_dict
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


_LABEL_COLS = [
    "atelectasis_left",
    "atelectasis_right",
    "heart_size",
    "pleural_effusion_left",
    "pleural_effusion_right",
    "pulmonary_congestion",
    "pulmonary_opacities_left",
    "pulmonary_opacities_right",
]


class _TAIXRayDatasetBase(BaseRadiologicalDataset):
    """Shared logic for the two TAIX-Ray dataset variants.

    Args:
        base_image_dir: Root of the PNG image tree (e.g.
            ``…/TAIX-Ray/data_512/images/`` or ``…/TAIX-Ray/data_original/images/``).
        csv_path: Path to ``annotation.csv``.  Auto-discovered near
            ``base_image_dir`` when ``None``.
        label_mode: ``"binary"`` (default) maps severity ≥ 1 → 1.0 for
            multi-label classification.  ``"ordinal"`` preserves raw 0–4
            grades for ordinal-regression heads.
        transform: MONAI Compose transform.  Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask / output_report / output_bbox: Not supported by TAIX-Ray
            (no masks, no reports, no bboxes); silently ignored with a warning.
        harmonized_df / harmonizer / harmonizer_path: Standard harmonizer
            preset hooks; see :class:`BaseRadiologicalDataset`.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})
    LABEL_COLS = _LABEL_COLS
    _HARMONIZER_CLS = TAIXRayHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_mode: str = "binary",
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
        if label_mode not in ("binary", "ordinal"):
            raise ValueError(
                f"label_mode must be 'binary' or 'ordinal', got {label_mode!r}"
            )
        self._label_mode = label_mode

        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if csv_path:
            csv_path = os.path.expanduser(csv_path)

        # Infer base_image_dir from a saved harmonizer if needed.
        if harmonizer_path is not None and base_image_dir is None:
            _h = TAIXRayHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_image_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        if output_mask:
            print("Warning: TAIX-Ray has no segmentation masks; output_mask=True ignored.")
        if output_report:
            print("Warning: TAIX-Ray has no radiology reports; output_report=True ignored.")
        if output_bbox:
            print("Warning: TAIX-Ray has no bounding boxes; output_bbox=True ignored.")

        # Auto-discover annotation.csv only when not loading from a preset.
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = (
                infer_path(base_image_dir, "annotation.csv", user_path=csv_path or "")
                or csv_path
            )
        else:
            self._csv_path = csv_path

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform or RadiologyTransform2D(
                output_keys={"img", "cls"} if output_cls else {"img"},
                dtype=dtype,
            ).get_transform(),
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=False,
            output_report=False,
            output_bbox=False,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            if not self._csv_path:
                raise FileNotFoundError(
                    f"Could not locate annotation.csv near {base_image_dir!r}; "
                    "pass csv_path= explicitly."
                )
            self._harmonizer = TAIXRayHarmonizer(csv_path=self._csv_path)

    def _apply_label_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._label_mode == "binary":
            for col in self.LABEL_COLS:
                if col in df.columns:
                    df[col] = (df[col].astype("float32") >= 1.0).astype("float32")
        return df

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        df = preset if preset is not None else self._harmonizer.harmonize()
        return self._apply_label_mode(df)

    def get_datasets_predefined(
        self,
        num_cores: int = 2,
        train_transform=None,
        val_transform=None,
        test_transform=None,
    ):
        """Return ``(train, val, test)`` per the paper's predefined patient-level split.

        Slices the harmonized DataFrame on its ``split`` column (values
        ``"train"`` / ``"val"`` / ``"test"``).  Each slice becomes a
        ``PersistentDataset`` with its own cache subdirectory.

        Raises:
            ValueError: If the harmonized DataFrame has no ``split`` column.

        Args:
            num_cores: Parallel workers for building data dicts.
            train_transform / val_transform / test_transform: Optional
                per-split MONAI transform overrides.

        Returns:
            ``(train_dataset, val_dataset, test_dataset)``.
        """
        df = self._get_harmonized_df()
        if "split" not in df.columns:
            raise ValueError(
                "Harmonized DataFrame has no 'split' column; cannot use predefined split. "
                "Use get_datasets(n_splits=...) for a random patient-level split instead."
            )
        self.LABEL_COLS = sorted(self.LABEL_COLS)
        cols = self._cols

        out = []
        for split_name, transform in (
            ("train", train_transform),
            ("val", val_transform),
            ("test", test_transform),
        ):
            sub = df[df["split"] == split_name]
            if sub.empty:
                raise ValueError(
                    f"Split '{split_name}' is empty in the harmonized DataFrame."
                )
            dicts = get_data_dict(sub, self.base_image_dir, "image_path", cols, num_cores)
            out.append(self._make_dataset(dicts, cache_subdir=split_name, transform=transform))
        return tuple(out)


@register_dataset("taix_ray_512")
class TAIXRay512Dataset(_TAIXRayDatasetBase):
    """TAIX-Ray with the 512-pixel resized PNG configuration.

    Use this for the standard ~58 GB release where the longer image
    dimension is uniformly 512 px (paper's default size for the
    Vision Transformer baseline).  Constructor args are identical to
    :class:`_TAIXRayDatasetBase`; ``base_image_dir`` should point at
    ``…/TAIX-Ray/data_512/images/``.
    """


@register_dataset("taix_ray")
class TAIXRayDataset(_TAIXRayDatasetBase):
    """TAIX-Ray with the original-resolution PNG configuration.

    Use this for the full ~1.27 TB release where each PNG preserves the
    native acquisition resolution.  Constructor args are identical to
    :class:`_TAIXRayDatasetBase`; ``base_image_dir`` should point at
    ``…/TAIX-Ray/data_original/images/``.
    """
