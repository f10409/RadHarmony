"""MONAI dataset for the ReXGradient-160K dataset."""

import pandas as pd
import torch

from radharmony.harmonizer import ReXGradientHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


_DEFAULT_JSON_VARIANTS = (
    "train_metadata_view_position.json",
    "valid_metadata_view_position.json",
    "test_metadata_view_position.json",
)


@register_dataset("rexgradient")
class ReXGradientDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for ReXGradient-160K.

    273,004 chest X-rays from 160,000 studies / 109,487 patients across
    79 U.S. medical sites, distributed via HuggingFace by rajpurkarlab.
    Three splits: train (238,968 images), validation (17,007), public
    test (17,029). Per-study text reports are pre-segmented into
    Indication / Comparison / Findings / Impression sections; the
    harmonizer concatenates them with section headers into the
    ``report`` field. There are no structured pathology labels in this
    dataset.

    The source metadata is the per-split JSON file
    (``*_metadata_view_position.json``), which carries the per-image
    ``ImagePath`` lists; the per-study CSV that ships alongside is not
    sufficient on its own. The ``csv_path`` argument is kept for
    interface compatibility with the rest of RadHarmony — it must point
    at the JSON file. Passing a ``.csv`` path raises ``ValueError``.

    Args:
        base_image_dir: Root of the extracted PNG tree (e.g.
            ``.../ReXGradient-160K/deid_png/``). Immediate children are
            ``<PatientID>/`` directories.
        csv_path: Path to ``<split>_metadata_view_position.json``.
            Auto-inferred from sibling/ancestor ``metadata/`` directories
            when ``None``. The arg name is ``csv_path`` for consistency
            with other RadHarmony datasets, but the value must be a
            ``.json`` path.
        transform: MONAI Compose transform. Defaults to the standard
            2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables
            caching.
        output_cls: ReXGradient ships no classification labels; setting
            ``True`` emits a ``UserWarning`` via the base class.
        output_report: Yield the concatenated 4-section report text
            under key ``report``.
        output_mask: Not supported.
        output_bbox: Not supported here (a small interstitial-pattern
            bbox JSON exists but is out of scope for this dataset class).
        dtype: Output tensor dtype for the default transform.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"report"})
    _HARMONIZER_CLS = ReXGradientHarmonizer
    LABEL_COLS: list = []

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
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
        if harmonizer_path is not None and base_image_dir is None:
            _h = ReXGradientHarmonizer.load_from_saved(harmonizer_path)
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

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = (
                infer_path(
                    base_image_dir,
                    *self._json_filename_variants(),
                    user_path=csv_path or "",
                )
                or csv_path
            )
            if self._csv_path is None:
                raise ValueError(
                    "ReXGradient metadata JSON not found near "
                    f"{base_image_dir!r}. Pass csv_path pointing at "
                    "'<split>_metadata_view_position.json' explicitly."
                )
            if str(self._csv_path).lower().endswith(".csv"):
                raise ValueError(
                    "ReXGradient requires the per-image JSON metadata "
                    "(e.g. 'train_metadata_view_position.json'); the "
                    "per-study CSV does not contain image paths. Got: "
                    f"{self._csv_path!r}"
                )
        else:
            self._csv_path = csv_path

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_report:
                output_keys.add("report")
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

    @classmethod
    def _json_filename_variants(cls) -> tuple[str, ...]:
        """Filenames :func:`infer_path` should try when csv_path is omitted.

        The base class is split-agnostic — it accepts any of the three
        split JSONs. Split-specific subclasses (Train/Valid/Test) tighten
        this to their own file so the dataset binds to the right split.
        """
        return _DEFAULT_JSON_VARIANTS

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = ReXGradientHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()
