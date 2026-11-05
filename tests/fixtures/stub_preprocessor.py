"""A stand-in preprocessor for the tests of the copy step. It writes nothing."""
from __future__ import annotations


class StubPreprocessor:
    """Keep the table and the image folder of each ``run_batch`` call.

    Attributes:
        tables: a copy of the table of each call.
        folders: the ``base_image_dir`` of each call.
    """

    def __init__(self):
        self.tables = []
        self.folders = []

    @property
    def calls(self) -> int:
        """The number of ``run_batch`` calls."""
        return len(self.tables)

    def run_batch(self, df, base_output_dir, base_image_dir=None, num_workers=1):
        self.tables.append(df.copy())
        self.folders.append(base_image_dir)
