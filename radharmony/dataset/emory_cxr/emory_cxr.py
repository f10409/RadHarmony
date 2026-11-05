"""MONAI dataset for the Emory CXR v2 (internal) dataset."""

import pandas as pd
import torch

from radharmony.harmonizer import EmoryCXRHarmonizer, EmoryCXRDicomHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base import BaseRadiologicalDataset
from ..transforms import RadiologyTransform2D


#: Metadata manifest filenames, newest first.  :func:`infer_path` tries
#: them in order. A new manifest in ``TABLES/`` is not used until you add it
#: here. This is on purpose, because the manifests are not interchangeable
#: (see :class:`EmoryCXRDataset`).
_METADATA_VARIANTS = (
    "EmoryCXR_v2_Metadata_08152025.csv",
    "EmoryCXR_v2_Metadata_08142025.csv",
    "EmoryCXR_v2_Metadata_08112025.csv",
    "EmoryCXR_v2_Metadata_10172024.csv",
)

_LABEL_VARIANTS = ("EmoryCXR_v2_FindingLabel_10162024.csv",)

_REPORT_VARIANTS = ("EmoryCXR_v2_Report_05102026.csv",)


def _in_tables(names: tuple[str, ...]) -> tuple[str, ...]:
    """Return the names in ``TABLES/``, the table folder next to the image folder (D-35)."""
    return tuple(f"TABLES/{n}" for n in names)


@register_dataset("emory_cxr")
class EmoryCXRDataset(BaseRadiologicalDataset):
    """2-D MONAI PersistentDataset for Emory CXR v2.

    2,430,209 chest radiographs from 1,214,535 studies and 402,809
    patients, acquired 2008-08-30 through 2021-12-24 across 11
    anonymised institutions.  Internal Emory corpus, not public, and
    not redistributable.

    Every study has a free-text report, except 3 studies (7 images) with
    only a blank template. ``output_report=True`` drops them (D-33). This
    makes the dataset useful as a report-generation target.  Reports
    average 66 words and carry an IMPRESSION in 98.9% of cases, FINDINGS
    in 86.5%.  The 14 CheXpert-style finding labels are available but are
    **sparse**, blanks dominate every column and mean "unmentioned", not
    "absent".

    Images are native-resolution 16-bit grayscale PNGs (``I;16``) at
    acquisition size: commonly 3032x2520 or 2021x2021, not pre-resized.
    Intensities are per-image min-max normalized to the full 16-bit
    range with no VOI LUT applied, so they are not comparable across
    images without renormalisation.

    Splits are the caller's responsibility, this dataset ships none.
    Split on ``patient_id``: 42.7% of patients contribute more than one
    study (up to 296), so a study-level split leaks.

    Args:
        base_image_dir: Root of the PNG tree, for example
            ``/data/EmoryCXRv2/DEID_PNG/``.
            Immediate children are ``<empi_anon>/`` directories.
        csv_path: Path to ``EmoryCXR_v2_Metadata_<date>.csv``.
            Auto-inferred from a sibling/ancestor ``TABLES/`` directory
            when ``None``.
        label_csv_path: Path to ``EmoryCXR_v2_FindingLabel_<date>.csv``.
            Auto-inferred. Only read when ``output_cls=True``.
        report_csv_path: Path to ``EmoryCXR_v2_Report_<date>.csv``.
            Auto-inferred. Only read when ``output_report=True``.
        transform: MONAI Compose transform.  Defaults to the standard
            2-D pipeline.
        cache_dir: PersistentDataset cache directory.  ``None`` disables
            caching.
        output_cls: Yield the 14 finding labels under key ``cls``.  NaN
            (unmentioned) and -1 (uncertain) are preserved as-is. Decide
            a policy before training on them.
        output_report: Yield the report text under key ``report``.
        output_mask: Not supported.
        output_bbox: Not supported.
        dtype: Dtype of ``img`` in the default transform. The labels stay float32.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"cls", "report"})
    _HARMONIZER_CLS = EmoryCXRHarmonizer
    # The table columns are snake_case. Sorted, so LABEL_COLS[j] names cls[j] (D-05).
    LABEL_COLS: list = sorted(c.lower().replace(" ", "_") for c in EmoryCXRHarmonizer.LABEL_COLS)

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        label_csv_path: str = None,
        report_csv_path: str = None,
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
        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
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
                    *_METADATA_VARIANTS,
                    *_in_tables(_METADATA_VARIANTS),
                    user_path=csv_path or "",
                )
                or csv_path
            )
            if self._csv_path is None:
                raise FileNotFoundError(  # D-35
                    "Emory CXR metadata CSV not found near "
                    f"{base_image_dir!r}. Pass csv_path pointing at "
                    "'EmoryCXR_v2_Metadata_<date>.csv' explicitly. Note the "
                    "manifests are NOT interchangeable: 08112025 has "
                    "2,486,502 images / 410,449 patients, 08152025 has "
                    "2,430,209 / 402,809."
                )
            self._label_csv_path = (
                infer_path(
                    base_image_dir,
                    *_LABEL_VARIANTS,
                    *_in_tables(_LABEL_VARIANTS),
                    user_path=label_csv_path or "",
                )
                or label_csv_path
            ) if output_cls else None
            self._report_csv_path = (
                infer_path(
                    base_image_dir,
                    *_REPORT_VARIANTS,
                    *_in_tables(_REPORT_VARIANTS),
                    user_path=report_csv_path or "",
                )
                or report_csv_path
            ) if output_report else None

            if output_cls and self._label_csv_path is None:
                raise FileNotFoundError(  # D-35
                    "output_cls=True but the finding-label CSV was not found "
                    f"near {base_image_dir!r}. Pass label_csv_path pointing "
                    "at 'EmoryCXR_v2_FindingLabel_<date>.csv'."
                )
            if output_report and self._report_csv_path is None:
                raise FileNotFoundError(  # D-35
                    "output_report=True but the report CSV was not found near "
                    f"{base_image_dir!r}. Pass report_csv_path pointing at "
                    "'EmoryCXR_v2_Report_<date>.csv'."
                )
        else:
            self._csv_path = csv_path
            self._label_csv_path = label_csv_path
            self._report_csv_path = report_csv_path

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
            output_reg=output_reg,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = EmoryCXRHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            label_csv_path=self._label_csv_path,
            report_csv_path=self._report_csv_path,
        )
        return harmonizer.harmonize()


_MAPPING_VARIANTS = ("mapping_deid_dcm_sopid.csv",)


@register_dataset("emory_cxr_dicom")
class EmoryCXRDicomDataset(EmoryCXRDataset):
    """Emory CXR v2 served from the de-identified DICOM tree.

    .. warning::

       **8-bit pixels.** All five MD.ai batch manifests report
       ``BitsStored = 8`` and ``MONOCHROME2``, against the 16-bit PNGs in
       ``DEID_PNG/``. This tree is a re-encoded annotation export, not the
       acquisition source. For training pixels prefer
       :class:`EmoryCXRDataset`. Use this one when you need DICOM headers
       or the MD.ai annotation round-trip.

    Identical to :class:`EmoryCXRDataset` in every other respect, same
    labels, same reports, same patient/study keys, same split advice. The
    one extra requirement is ``mapping_csv_path``, since the DICOM files
    carry no cohort identifiers of their own.

    Args:
        base_image_dir: Root of the DICOM tree, for example
            ``<EmoryCXRv2>/DEID_DICOM/``. Children are ``B1/`` … ``B5/``.
        mapping_csv_path: Path to ``mapping_deid_dcm_sopid.csv``.
            Auto-inferred from a sibling/ancestor ``TABLES/`` directory.
        **kwargs: Forwarded to :class:`EmoryCXRDataset`.
    """

    _HARMONIZER_CLS = EmoryCXRDicomHarmonizer

    def __init__(self, base_image_dir: str = None, mapping_csv_path: str = None, **kwargs):
        self._mapping_csv_path = (
            infer_path(
                base_image_dir,
                *_MAPPING_VARIANTS,
                *_in_tables(_MAPPING_VARIANTS),
                user_path=mapping_csv_path or "",
            )
            or mapping_csv_path
        )
        super().__init__(base_image_dir=base_image_dir, **kwargs)

        if (
            self._mapping_csv_path is None
            and kwargs.get("harmonizer_path") is None
            and kwargs.get("harmonized_df") is None
            and kwargs.get("harmonizer") is None
        ):
            raise ValueError(
                "mapping_deid_dcm_sopid.csv not found near "
                f"{base_image_dir!r}. It is the only route from a .dcm "
                "back to a patient (the MD.ai export blanked "
                "AccessionNumber and regenerated PatientID), so pass "
                "mapping_csv_path explicitly."
            )

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        harmonizer = EmoryCXRDicomHarmonizer(
            csv_path=self._csv_path,
            base_image_dir=self.base_image_dir,
            mapping_csv_path=self._mapping_csv_path,
            label_csv_path=self._label_csv_path,
            report_csv_path=self._report_csv_path,
        )
        return harmonizer.harmonize()
