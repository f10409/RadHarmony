"""MONAI dataset for the MIMIC-CXR v2 dataset (DICOM source, supports reports)."""

import pandas as pd
import torch

from radharmony.harmonizer import MIMICCXRHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("mimic_cxr")
class MIMICCXRDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for MIMIC-CXR v2.

    Args:
        base_image_dir: Root of the image file tree, e.g.
            ``.../mimic-cxr/2.1.0/files/``.
        dicom_base_dir: Root of the DICOM file tree used to read
            ``ViewPosition`` from DICOM headers.  Defaults to
            ``base_image_dir`` when ``None``.
        csv_path: Path to the metadata CSV (``mimic-cxr-2.0.0-metadata.csv``
            or ``cxr-record-list.csv.gz``).  Must contain a ``path`` column.
            Auto-inferred near ``base_image_dir`` when ``None``.
        label_csv_path: Path to ``mimic-cxr-2.0.0-chexpert.csv`` (optional).
        report_csv_path: Path to ``cxr-study-list.csv.gz`` (optional).
        transform: MONAI Compose transform. Defaults to RadiologyTransform2D.
        cache_dir: PersistentDataset cache directory. None disables caching.
        output_cls: Yield label vector under key ``cls``.
        output_mask: Not supported; setting True has no effect.
        output_report: Yield report path under key ``report``.
            Requires ``report_csv_path``.
        output_bbox: Not supported; setting True has no effect.
        drop_uncertain: If ``True`` (default), rows with uncertain labels are dropped.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "report"})
    _HARMONIZER_CLS = MIMICCXRHarmonizer

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
        dicom_base_dir: str = None,
        csv_path: str = None,
        label_csv_path: str = None,
        report_csv_path: str = None,
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
        # When a saved harmonizer is provided, infer base_image_dir from its
        # stored report_base_dir so the caller need not repeat it.
        if harmonizer_path is not None and base_image_dir is None:
            _h = MIMICCXRHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = _h.report_base_dir

        if base_image_dir is None and harmonizer_path is None and harmonized_df is None and harmonizer is None:
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided."
            )

        # Only search for CSVs when not loading from a saved harmonizer.
        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._csv_path = (
                infer_path(
                    base_image_dir,
                    "cxr-record-list.csv.gz",
                    "cxr-record-list.csv",
                    "mimic-cxr-2.0.0-metadata.csv",
                    user_path=csv_path or "",
                )
                or csv_path
            )
            # Only search for the label CSV when it will actually be read
            # (output_cls=True or caller passed an explicit path). The full
            # infer_path search walks "uncle" directories under the dataset
            # root, which on NAS4 has descended into PMC-Rad-Plus and hung
            # on a degraded NFS readdir — skip it when labels aren't needed.
            if output_cls or label_csv_path:
                self._label_csv_path = (
                    infer_path(
                        base_image_dir,
                        "mimic-cxr-2.0.0-chexpert.csv",
                        "mimic-cxr-2.0.0-chexpert.csv.gz",
                        user_path=label_csv_path or "",
                    )
                    or label_csv_path
                )
            else:
                self._label_csv_path = None
            self._report_csv_path = (
                infer_path(
                    base_image_dir,
                    "cxr-study-list.csv.gz",
                    "cxr-study-list.csv",
                    user_path=report_csv_path or "",
                )
                or report_csv_path
            )
        else:
            self._csv_path = csv_path
            self._label_csv_path = label_csv_path
            self._report_csv_path = report_csv_path

        self._dicom_base_dir = dicom_base_dir
        self._drop_uncertain = drop_uncertain

        output_keys = {"img"}
        if output_cls:
            output_keys.add("cls")
        if output_mask:
            print(
                "Warning: MIMIC-CXR does not include segmentation masks; output_mask=True has no effect."
            )
        if output_report:
            output_keys.add("report")
        if output_bbox:
            print(
                "Warning: MIMIC-CXR does not include bounding boxes; output_bbox=True has no effect."
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

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return MIMICCXRHarmonizer(
            csv_path=self._csv_path,
            label_csv_path=self._label_csv_path,
            dicom_base_dir=self._dicom_base_dir,
            report_csv_path=self._report_csv_path,
            report_base_dir=self.base_image_dir,
        ).harmonize(drop_uncertain=self._drop_uncertain)
