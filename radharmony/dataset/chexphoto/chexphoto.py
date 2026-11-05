"""MONAI dataset for CheXphoto: smartphone / synthetic photographs of CheXpert x-rays.

See :class:`~radharmony.harmonizer.CheXphotoHarmonizer` for the three ways the
released data differs from a plain CheXpert clone:

* Underscored column names.
* The ``CheXphoto-v1.0/<split>/`` path prefix.
* No ``film`` subset.
"""

from radharmony.harmonizer import CheXphotoHarmonizer
from radharmony.registry import register_dataset

from ..chexpert.chexpert import CheXpertDataset


@register_dataset("chexphoto")
class CheXphotoDataset(CheXpertDataset):
    """2-D MONAI PersistentDataset for CheXphoto.

    Labels are CheXpert's 14 observations, unchanged.  The same source x-ray
    appears under several capture conditions. So ``subset`` and the
    ``capture_method`` column are the axes of interest.

    Args:
        base_image_dir: Split root containing ``natural/`` and ``synthetic/``.
        csv_path: ``train.csv`` or ``valid.csv``.  Defaults to ``train.csv``
            alongside ``base_image_dir``.
        subset: ``all`` (default), ``natural`` or ``synthetic``.
        drop_uncertain: As CheXpert: drop a row with ``-1`` in any of the 14 labels. This
            removes 12,487 of the 32,521 training rows. ``not_mentioned`` works as in
            :class:`CheXpertDataset` (D-05).
        Remaining arguments are inherited from :class:`CheXpertDataset`.
    """

    _HARMONIZER_CLS = CheXphotoHarmonizer
    _SPLIT_CSV = "train.csv"

    def __init__(self, base_image_dir: str = None, csv_path: str = None,
                 subset: str = "all", **kwargs):
        # Default the CSV off base_image_dir rather than a hardcoded root so
        # that pointing at another copy of the tree picks up that copy's CSV.
        if csv_path is None and base_image_dir:
            csv_path = f"{base_image_dir.rstrip('/')}/{self._SPLIT_CSV}"
        self.subset = subset
        super().__init__(base_image_dir=base_image_dir, csv_path=csv_path, **kwargs)
        # Swap in the CheXphoto harmonizer AFTER the parent runs.
        # CheXpertDataset hardcodes CheXpertHarmonizer in its own __init__ and
        # ignores _HARMONIZER_CLS. That harmonizer gives wrong column names
        # (see COL_MAP) and wrong patient ids ("synthetic"). This also pins the
        # split-specific subclass so save() / load_from_saved() round-trip.
        if kwargs.get("harmonizer") is None and \
                kwargs.get("harmonizer_path") is None and \
                kwargs.get("harmonized_df") is None:
            self._harmonizer = self._HARMONIZER_CLS(
                csv_path=csv_path, base_image_dir=base_image_dir, subset=subset)
