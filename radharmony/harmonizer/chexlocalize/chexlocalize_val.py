"""Harmonizer for the CheXlocalize validation split (the CheXpert valid set)."""

from .chexlocalize import CheXlocalizeHarmonizer


class CheXlocalizeValHarmonizer(CheXlocalizeHarmonizer):
    """CheXlocalize validation split: ``val_labels.csv`` and ``gt_segmentations_val.json``.

    The Path column is ``CheXpert-v1.0/valid/patient.../study.../view...jpg``.
    ``base_image_dir`` is the ``val/`` folder, so the class strips the part
    before the patient folder. The arguments are the same as
    :class:`CheXlocalizeHarmonizer`.
    """

    _PATH_PREFIX = r"^(CheXpert-v1\.0/)?valid?/"

    def _build_image_path(self) -> None:
        if not self.df["Path"].str.contains(self._PATH_PREFIX, regex=True).all():
            raise ValueError(f"{self.csv_path} has rows of another split. Pass val_labels.csv.")
        super()._build_image_path()
