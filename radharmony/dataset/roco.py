"""MONAI dataset for the ROCO (Radiology Objects in COntext) captioning dataset."""

import torch

from radharmony.harmonizer.roco import ROCOHarmonizer
from radharmony.registry import register_dataset
from .base_vqa import BaseVQADataset


@register_dataset("roco")
class ROCODataset(BaseVQADataset):
    """Captioning dataset built on ROCO (Radiology Objects in COntext).

    Each sample pairs a radiology image with its PubMed Central figure caption.
    ``question`` is always an empty string (pure captioning); ``answer`` is the
    caption text.  Optional ``keywords`` is a tab-joined keyword string derived
    from ROCO's ``keywords.txt``.

    Args:
        base_dir: Root of the roco-dataset clone/download tree — the directory
            containing the ``data/`` subdirectory.  Required unless
            ``harmonized_df`` / ``harmonizer`` / ``harmonizer_path`` is given.
        splits: Which dataset splits to load.  Defaults to all three:
            ``["train", "validation", "test"]``.
        radiology_only: When ``True`` (default), load only the ``radiology/``
            subset.  Set ``False`` to include ``non-radiology/`` images as well.
        image_subdir: Image subdirectory name inside each split/subset folder
            (default ``"images"``).  Must match the ``--subdir`` argument used
            when running ``scripts/fetch.py``.
        transform, cache_dir, output_struct, output_question_type:
            See :class:`~radharmony.dataset.base_vqa.BaseVQADataset`.
        harmonized_df, harmonizer, harmonizer_path: Standard harmonizer
            resolution paths; when used, ``base_dir`` is not required.
    """

    _HARMONIZER_CLS = ROCOHarmonizer

    def __init__(
        self,
        base_dir: str = None,
        splits: list[str] | None = None,
        radiology_only: bool = True,
        image_subdir: str = "images",
        transform=None,
        cache_dir: str = "./cache",
        output_struct: bool = False,
        output_question_type: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if harmonizer_path is not None and base_dir is None:
            _h = ROCOHarmonizer.load_from_saved(harmonizer_path)
            base_dir = _h.base_dir

        if (
            base_dir is None
            and harmonized_df is None
            and harmonizer is None
            and harmonizer_path is None
        ):
            raise ValueError(
                "base_dir (roco-dataset root containing data/) is required "
                "when harmonizer_path / harmonizer / harmonized_df are not provided."
            )

        super().__init__(
            base_image_dir=base_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_struct=output_struct,
            output_question_type=output_question_type,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
            dtype=dtype,
        )

        if harmonized_df is None and harmonizer is None and harmonizer_path is None:
            self._harmonizer = ROCOHarmonizer(
                base_dir=base_dir,
                splits=splits,
                radiology_only=radiology_only,
                image_subdir=image_subdir,
            )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()
