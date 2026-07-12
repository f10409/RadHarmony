"""VQA dataset for VQA-RAD (Visual Question Answering in Radiology)."""

import torch

from radharmony.harmonizer import VQARadHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base_vqa import BaseVQADataset


@register_dataset("vqa_rad")
class VQARadDataset(BaseVQADataset):
    """VQA dataset over VQA-RAD.

    Args:
        base_image_dir: ``VQA_RAD Image Folder/`` directory, e.g.
            ``/data/VQA-RAD/VQA_RAD Image Folder/``.
        json_path: Path to ``VQA_RAD Dataset Public.json``. Auto-discovered
            in the parent of ``base_image_dir`` when ``None``.
        transform: MONAI transform pipeline. Defaults to the standard VQA
            2-D pipeline (loads ``img``; text fields pass through unchanged).
        cache_dir: Root directory for MONAI PersistentDataset cache.
        output_struct: Include ``answer_struct`` in each sample (unset for
            VQA-RAD — its answers are plain strings, not structured).
        output_question_type: Include ``question_type`` in each sample.
        harmonized_df, harmonizer, harmonizer_path: Standard harmonizer
            resolution paths; when used, ``json_path`` is not required.
    """

    _HARMONIZER_CLS = VQARadHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        json_path: str = None,
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
            _h = VQARadHarmonizer.load_from_saved(harmonizer_path)
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

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._json_path = infer_path(
                base_image_dir, "VQA_RAD Dataset Public.json", user_path=json_path or "",
            ) or json_path
        else:
            self._json_path = json_path

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

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._harmonizer = VQARadHarmonizer(
                json_path=self._json_path,
                base_image_dir=base_image_dir,
            )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()
