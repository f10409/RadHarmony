"""MONAI dataset for the NIH ChestX-ray14 dataset (non-bbox images)."""

import warnings

import pandas as pd
import torch

from radharmony.harmonizer import ChestXray14Harmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("chestxray14")
class ChestXray14Dataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for NIH ChestX-ray14 (non-bbox images).

    ~111,240 frontal-view chest X-rays (PNG) that do **not** have
    bounding-box annotations.  For the ~880 images with bounding boxes,
    use :class:`ChestXray14BboxDataset`.

    Images are distributed across ``images_001/images/`` …
    ``images_012/images/`` under ``base_image_dir``
    (e.g. ``.../NIH_CXR/CXR14/``).

    Labels are NLP-derived from radiology reports (15 classes including
    ``No Finding``).

    Args:
        base_image_dir: Root directory containing ``images_*/images/`` folders
            (e.g. ``.../NIH_CXR/CXR14/``).
        csv_path: Path to ``Data_Entry_2017.csv``.  Auto-inferred when ``None``.
        bbox_csv_path: Path to ``BBox_List_2017.csv`` (used to exclude bbox images).
            Auto-inferred when ``None``.
        transform: MONAI Compose transform. Defaults to standard 2-D pipeline.
        cache_dir: PersistentDataset cache directory. ``None`` disables caching.
        output_cls: Yield label vector under key ``cls``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls"})
    LABEL_COLS = [
        "atelectasis",
        "cardiomegaly",
        "consolidation",
        "edema",
        "effusion",
        "emphysema",
        "fibrosis",
        "hernia",
        "infiltration",
        "mass",
        "no_finding",
        "nodule",
        "pleural_thickening",
        "pneumonia",
        "pneumothorax",
    ]
    _HARMONIZER_CLS = ChestXray14Harmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        bbox_csv_path: str = None,
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
            _h = ChestXray14Harmonizer.load_from_saved(harmonizer_path)
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
            warnings.warn(
                "ChestX-ray14 does not provide segmentation masks. "
                "output_mask will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            output_mask = False
        if output_report:
            warnings.warn(
                "ChestX-ray14 does not provide radiology reports. "
                "output_report will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            output_report = False
        if output_bbox:
            warnings.warn(
                "ChestX-ray14 (non-bbox split) does not include bounding boxes. "
                "Use ChestXray14BboxDataset instead. output_bbox will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            output_bbox = False

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = (
                infer_path(
                    base_image_dir,
                    "Data_Entry_2017.csv",
                    user_path=csv_path or "",
                )
                or csv_path
            )
            self._bbox_csv_path = (
                infer_path(
                    base_image_dir,
                    "BBox_List_2017.csv",
                    user_path=bbox_csv_path or "",
                )
                or bbox_csv_path
            )
        else:
            self._csv_path = csv_path
            self._bbox_csv_path = bbox_csv_path

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
        harmonizer = ChestXray14Harmonizer(
            csv_path=self._csv_path,
            bbox_csv_path=self._bbox_csv_path,
            base_image_dir=self.base_image_dir,
        )
        return harmonizer.harmonize()

    def verify_images(self, drop_missing: bool = True) -> pd.DataFrame:
        """Check that every ``image_path`` exists on disk, is single-channel,
        and optionally drop bad rows.

        Rows are dropped if the file is missing **or** the image has more than
        1 channel (e.g. RGB PNGs that slipped into the dataset).

        Args:
            drop_missing: If ``True`` (default), bad rows are removed from the
                harmonized DataFrame.

        Returns:
            DataFrame of dropped rows (empty if all files are valid).
        """
        import os

        from PIL import Image
        from tqdm import tqdm

        df = self._get_harmonized_df()

        def _check(p):
            fp = os.path.join(self.base_image_dir, p) if self.base_image_dir else p
            if not os.path.isfile(fp):
                return "missing"
            try:
                with Image.open(fp) as im:
                    channels = len(im.getbands())
                if channels > 1:
                    return "multichannel"
            except Exception:
                return "unreadable"
            return "ok"

        tqdm.pandas(desc="Verifying images")
        status = df["image_path"].progress_apply(_check)
        bad_mask = status != "ok"
        bad_df = df[bad_mask].copy()
        bad_df["verify_reason"] = status[bad_mask].values

        n_missing = (status == "missing").sum()
        n_multi = (status == "multichannel").sum()
        n_unread = (status == "unreadable").sum()

        if not bad_df.empty and drop_missing:
            kept = df[~bad_mask].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(
                f"Dropped {len(bad_df)} rows "
                f"(missing={n_missing}, multichannel={n_multi}, unreadable={n_unread}), "
                f"{len(kept)} remaining."
            )
        elif bad_df.empty:
            print("All image files valid (single-channel).")
        else:
            print(
                f"Found {len(bad_df)} bad rows "
                f"(missing={n_missing}, multichannel={n_multi}, unreadable={n_unread}), "
                f"not dropped."
            )

        return bad_df
