"""Harmonizer for a custom (toy) copy of the Montgomery County TB CXR set.

Datathon example of the *CSV-driven* harmonizer pattern: the raw dataset is a
folder of PNG chest X-rays plus a small metadata CSV
(``data/montgomery_index.csv``) with one row per image and columns
``pid`` / ``fname`` / ``tb``.

Raw CSV columns:
    pid    patient identifier (one study per patient here)
    fname  image filename, relative to base_image_dir
    tb     tuberculosis label (0 = normal, 1 = TB-positive)

Harmonized columns produced:
    patient_id, study_id, image_path, tb

The default :meth:`BaseHarmonizer.harmonize` flow already reads ``csv_path``
into ``self.df`` and calls the ``_build_*`` hooks, so this harmonizer only
overrides the three that map raw columns onto the canonical ones. ``tb`` is
already a column, so ``LABEL_COLS`` alone carries it into the output.
"""

import pandas as pd

from ..base import BaseHarmonizer


class MyMontgomeryHarmonizer(BaseHarmonizer):
    """Harmonize the toy Montgomery CSV into the standard RadHarmony format."""

    # Original CSV column names. The base class snake-cases them and keeps
    # them in the harmonized slice. Must be sorted alphabetically.
    LABEL_COLS = ["tb"]

    def _build_patient_id(self) -> None:
        # pid is the patient identifier; cast to str for stable grouping.
        self.df["patient_id"] = self.df["pid"].astype(str)

    def _build_study_id(self) -> None:
        # One study per patient in this set, so study_id mirrors patient_id.
        self.df["study_id"] = self.df["pid"].astype(str)

    def _build_image_path(self) -> None:
        # fname is already a bare filename relative to base_image_dir.
        self.df["image_path"] = self.df["fname"].astype(str)
