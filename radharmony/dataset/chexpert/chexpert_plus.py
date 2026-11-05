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

    The table keeps rows whose DICOM file is missing. A copy of the release
    can miss some patient folders, and the harmonizer then gives a warning.
    Call :meth:`verify_images` to drop those rows before you iterate.

    Args:
        base_image_dir: The ``DICOM/Uncompressed/`` directory inside the
            CheXpert-Plus release, for example
            ``.../chexpertplus/DICOM/Uncompressed/``.
            ``df_chexpert_plus_240401.csv`` and ``impression_fixed.json``
            are auto-discovered near this directory.
        csv_path: Path to ``df_chexpert_plus_240401.csv``. Found in
            ``base_image_dir``, its parent or its grandparent when ``None``.
        label_json_path: Path to ``impression_fixed.json`` (JSONL label file, D-48).
            Found in the same places when ``None``. Needed for
            ``output_cls=True`` (D-35).
        transform: MONAI Compose transform.  Defaults to RadiologyTransform2D.
        cache_dir: PersistentDataset cache directory.  ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Not supported. Setting ``True`` has no effect.
        output_report: Yield inline report text under key ``report``.
        output_bbox: Not supported. Setting ``True`` has no effect.
        drop_uncertain: Drop the rows with an uncertain label (``-1``). With ``False``, ``-1``
            stays in ``cls`` (D-05).
        not_mentioned: ``"negative"`` (default) gives 0 for a blank label, ``"nan"`` keeps NaN (D-05).
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
        output_reg: bool = False,
        output_bbox: bool = False,
        drop_uncertain: bool = True,
        not_mentioned: str = "negative",
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
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
            # The CSV and the JSON are in the chexpertplus root, up to 2 folders
            # above base_image_dir (for example .../PNG/train/). The "../" name
            # adds the grandparent to the exact places (D-35).
            self._csv_path = infer_path(
                base_image_dir,
                "df_chexpert_plus_240401.csv", "../df_chexpert_plus_240401.csv",
                user_path=csv_path or "",
                required=True,
                arg="csv_path",
            )
            # The labels come from impression_fixed.json (D-48), so output_cls needs it.
            self._label_json_path = infer_path(
                base_image_dir,
                "impression_fixed.json", "../impression_fixed.json",
                user_path=label_json_path or "",
                required=output_cls,
                arg="label_json_path",
            ) or None
        else:
            self._csv_path = csv_path
            self._label_json_path = label_json_path
        self._drop_uncertain = drop_uncertain
        self._not_mentioned = not_mentioned

        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")
        if output_report:
            output_keys.add("report")

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
            output_reg=output_reg,
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
        return self._HARMONIZER_CLS(
            csv_path=self._csv_path,
            label_json_path=self._label_json_path,
            base_image_dir=self.base_image_dir,
        ).harmonize(drop_uncertain=self._drop_uncertain, not_mentioned=self._not_mentioned)
