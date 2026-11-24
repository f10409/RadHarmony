"""Age rule shared by the VinDr-CXR train and test harmonizers."""

import numpy as np
import pandas as pd


class _AgeZeroIsUnknownMixin:
    """A DICOM ``PatientAge`` of 0 (``000Y`` or ``000D``) gives NaN (D-06).

    In VinDr-CXR, 0 means "unknown". The tags give 0 for 905 train and 269
    test images. Each age from 2 to 17 has only 2 to 12 images. Put this class
    before ``BaseHarmonizer`` in the class list.
    """

    def _build_age_years(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._build_age_years(df)
        if "age_years" in df.columns:
            df.loc[df["age_years"] == 0, "age_years"] = np.nan
        return df
