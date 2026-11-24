"""MONAI dataset for PadChest-GR: grounded reports on PadChest radiographs.

See :class:`~radharmony.harmonizer.PadChestGRHarmonizer` for the seven traps in
the release:

* The trailing-space directory.
* The split zip.
* The ``[x0, y0, x1, y1]`` box order.
* No box does not mean no finding.
* One master-table row per finding.
* Multi-label sentences.
* The undocumented ``extra_boxes``.

The 4,555 images are PadChest images, so this overlaps ``padchest`` completely.
"""
import os

import pandas as pd
import torch

from radharmony.harmonizer import PadChestGRHarmonizer
from radharmony.harmonizer.padchest_gr import LABEL_COLUMNS
from radharmony.registry import register_dataset

from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


@register_dataset("padchest_gr")
class PadChestGRDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for PadChest-GR.

    4,555 chest radiographs (PA / AP) with sentence-level grounded reports.
    Each has a 154-label multi-hot vector, boxes (7,733 in all) tied to the
    sentence that describes them, and the report in English or Spanish.

    Args:
        base_image_dir: The release directory. It holds
            ``grounded_reports_*.json``, ``master_table.csv`` and ``PadChest_GR/``.
        csv_path: ``master_table.csv``. Defaults to the one in ``base_image_dir``.
        json_path: ``grounded_reports_20240819.json``. Defaults likewise.
        padchest_csv_path: PadChest's own label CSV (``.csv.gz``). When given,
            ``view_position`` is filled from its ``Projection`` column.
        split: ``train`` / ``validation`` / ``test`` to keep one official split
            (patient-disjoint), or ``None`` for all 4,555.
        report_language: ``en`` (default) or ``es``.
        output_cls: Yield the 154-element label vector under ``cls``.
        output_report: Yield the report under ``report``.
        output_bbox: Yield boxes as normalized
            ``[dim0_min, dim0_max, dim1_min, dim1_max]`` under ``bbox``.
            ``bbox_labels`` holds the finding's label (``" + "``-joined when a
            sentence has several). ``bbox_findings`` holds its English sentence.
            Both are parallel to the boxes.
        output_mask: Not supported. A warning, and no key (D-11).

    In the table, ``bbox``, ``bbox_labels`` and ``bbox_findings`` are JSON-encoded
    strings, because per-image lists are ragged. An image with no box has ``"[]"``
    for each, as NODE21's controls do. In a sample, ``bbox`` is an ``(N, 4)`` tensor
    and the other 2 are lists (D-28). ``n_boxes == 0`` does not mean no finding:
    91 images with an abnormal finding have no box.

    The pixels are PadChest's, so a GR test patient is very probably a ``padchest``
    training patient: do not pair the two without removing them.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "bbox", "report"})
    SUPPORTS_BBOX_FINDINGS: bool = True
    LABEL_COLS = LABEL_COLUMNS
    _HARMONIZER_CLS = PadChestGRHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        json_path: str = None,
        padchest_csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
        split: str = None,
        report_language: str = "en",
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
        output_mask: bool = False,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)  # as the harmonizer does
        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "base_image_dir", None)

        preset = harmonizer_path is not None or harmonized_df is not None or harmonizer is not None
        if base_image_dir is None and not preset:
            raise ValueError(
                "base_image_dir is required when harmonizer_path is not provided.")

        self._csv_path = csv_path
        self._json_path = json_path
        self._padchest_csv_path = padchest_csv_path
        self._split = split
        self._report_language = report_language

        if transform is None:
            output_keys = {"img"}
            if output_cls:
                output_keys.add("cls")
            if output_report:
                output_keys.add("report")
            if output_bbox:
                output_keys.update({"bbox", "bbox_labels", "bbox_findings"})
            transform = RadiologyTransform2D(
                img_size=224, output_keys=output_keys, dtype=dtype,
            ).get_transform()

        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=output_bbox,
            output_mask=output_mask,  # not in SUPPORTED_OUTPUTS: the base class warns
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._HARMONIZER_CLS(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            json_path=self._json_path,
            padchest_csv_path=self._padchest_csv_path,
            split=self._split,
            report_language=self._report_language,
        ).harmonize()
