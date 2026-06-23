"""MONAI dataset for the CheXpert-Plus dataset (DICOM source, inline reports)."""

import pandas as pd
import torch

from radharmony.harmonizer import CheXpertPlusHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from radharmony.utils.data_utils import get_data_dict
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("chexpert_plus")
class CheXpertPlusDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for CheXpert-Plus (DICOM).

    CheXpert-Plus combines CheXpert-style classification labels with
    full radiology reports (inline) and DICOM image files, matching the
    study population of the Stanford CheXpert dataset.

    Args:
        base_image_dir: The ``DICOM/Uncompressed/`` directory inside the
            CheXpert-Plus release, e.g.
            ``.../chexpertplus/DICOM/Uncompressed/``.
            ``df_chexpert_plus_240401.csv`` and ``report_fixed.json``
            are auto-discovered near this directory.
        csv_path: Path to ``df_chexpert_plus_240401.csv``.  Auto-inferred
            near ``base_image_dir`` when ``None``.
        label_json_path: Path to ``report_fixed.json`` (JSONL label file).
            Auto-inferred near ``base_image_dir`` when ``None``.
        transform: MONAI Compose transform.  Defaults to RadiologyTransform2D.
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Not supported; setting ``True`` has no effect.
        output_report: Yield inline report text under key ``report``.
        output_bbox: Not supported; setting ``True`` has no effect.
        drop_uncertain: Drop rows where any label is ``null`` (uncertain).
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "report"})
    _HARMONIZER_CLS = CheXpertPlusHarmonizer

    LABEL_COLS = [
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

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_json_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_bbox: bool = False,
        drop_uncertain: bool = True,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if harmonizer_path is not None and base_image_dir is None:
            _h = CheXpertPlusHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = _h.base_image_dir

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
            # CSV and JSON live in the chexpertplus root, which may be
            # several levels above base_image_dir (e.g. .../DICOM/Uncompressed/).
            # Search from base_image_dir; fall back to walking up explicitly.
            self._csv_path = csv_path or infer_path(
                base_image_dir,
                "df_chexpert_plus_240401.csv",
                user_path=csv_path or "",
            )
            self._label_json_path = label_json_path or infer_path(
                base_image_dir,
                "impression_fixed.json",
                user_path=label_json_path or "",
            )
            if not self._csv_path or not self._label_json_path:
                import os

                d = os.path.normpath(base_image_dir)
                for _ in range(4):
                    d = os.path.dirname(d)
                    if not self._csv_path:
                        p = os.path.join(d, "df_chexpert_plus_240401.csv")
                        if os.path.exists(p):
                            self._csv_path = p
                    if not self._label_json_path:
                        p = os.path.join(d, "report_fixed.json")
                        if os.path.exists(p):
                            self._label_json_path = p
                    if self._csv_path and self._label_json_path:
                        break
        else:
            self._csv_path = csv_path
            self._label_json_path = label_json_path
        self._drop_uncertain = drop_uncertain

        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")
        if output_mask:
            print(
                "Warning: CheXpert-Plus does not include segmentation masks; "
                "output_mask=True has no effect."
            )
        if output_report:
            output_keys.add("report")
        if output_bbox:
            print(
                "Warning: CheXpert-Plus does not include bounding boxes; "
                "output_bbox=True has no effect."
            )

        if transform is None:
            transform = RadiologyTransform2D(
                img_size=224, output_keys=output_keys,
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

    def get_original_splits(self, num_cores: int = 2):
        """Return (train_dataset, val_dataset) using the official ``split`` column.

        Uses the pre-defined ``train`` / ``valid`` split from
        ``df_chexpert_plus_240401.csv`` instead of a random patient split.

        Returns:
            Tuple of (train_dataset, val_dataset).

        Raises:
            ValueError: If the harmonized DataFrame has no ``split`` column.
        """
        df = self._get_harmonized_df()
        if "split" not in df.columns:
            raise ValueError(
                "No 'split' column found. Ensure the harmonizer preserved it "
                "(re-harmonize without a stale pkl)."
            )
        cols = self._cols
        train_dicts = get_data_dict(
            df[df["split"] == "train"].reset_index(drop=True),
            self.base_image_dir,
            "image_path",
            cols,
            num_cores,
        )
        val_dicts = get_data_dict(
            df[df["split"] == "valid"].reset_index(drop=True),
            self.base_image_dir,
            "image_path",
            cols,
            num_cores,
        )
        return self._make_dataset(train_dicts, "train"), self._make_dataset(
            val_dicts, "val"
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return CheXpertPlusHarmonizer(
            csv_path=self._csv_path,
            label_json_path=self._label_json_path,
            base_image_dir=self.base_image_dir,
        ).harmonize(drop_uncertain=self._drop_uncertain)
