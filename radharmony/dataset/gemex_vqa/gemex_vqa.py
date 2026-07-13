"""Dataset for GEMeX-VQA (visual question-answering over MIMIC-CXR images)."""

import torch

from radharmony.harmonizer import GEMeXVQAHarmonizer
from radharmony.registry import register_dataset
from ..base_vqa import BaseVQADataset


@register_dataset("gemex_vqa")
class GEMeXVQADataset(BaseVQADataset):
    """VQA dataset built on GEMeX-VQA annotations over MIMIC-CXR-JPG images.

    Args:
        data_dir: Directory containing the 4 GEMeX-VQA JSONL files
            downloaded from https://huggingface.co/datasets/BoKelvin/GEMeX-VQA.
        base_image_dir: MIMIC-CXR-JPG ``files/`` root, e.g.
            ``/data/mimic-cxr-jpg/2.0.0/files``.  Image paths in the
            harmonized DataFrame are relative to this directory.
        question_subtypes: Which subtypes to include.  Defaults to all four:
            ``["open_ended", "closed_ended", "single_choice", "multi_choice"]``.
        transform, cache_dir, output_struct, output_question_type:
            See :class:`~radharmony.dataset.base_vqa.BaseVQADataset`.
        harmonized_df, harmonizer, harmonizer_path: Standard harmonizer
            resolution paths; when used, ``data_dir`` is not required.
    """

    _HARMONIZER_CLS = GEMeXVQAHarmonizer

    def __init__(
        self,
        data_dir: str = None,
        base_image_dir: str = None,
        question_subtypes: list[str] | None = None,
        transform=None,
        cache_dir: str = "./cache",
        output_struct: bool = False,
        output_question_type: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if harmonizer_path is not None and base_image_dir is None:
            _h = GEMeXVQAHarmonizer.load_from_saved(harmonizer_path)
            base_image_dir = _h.base_image_dir

        if (
            base_image_dir is None
            and harmonized_df is None
            and harmonizer is None
            and harmonizer_path is None
        ):
            raise ValueError(
                "base_image_dir is required when harmonizer_path / harmonizer / "
                "harmonized_df are not provided."
            )

        super().__init__(
            base_image_dir=base_image_dir,
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
            if data_dir is None:
                raise ValueError(
                    "data_dir is required when constructing the harmonizer from scratch."
                )
            self._harmonizer = GEMeXVQAHarmonizer(
                data_dir=data_dir,
                base_image_dir=base_image_dir,
                question_subtypes=question_subtypes,
            )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()
