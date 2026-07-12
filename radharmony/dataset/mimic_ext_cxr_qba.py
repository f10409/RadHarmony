"""Dataset for MIMIC-Ext-CXR-QBA (visual question-answering over MIMIC-CXR)."""

import torch

from radharmony.harmonizer.mimic_ext_cxr_qba import MIMICExtCXRQBAHarmonizer
from radharmony.registry import register_dataset
from .base_vqa import BaseVQADataset


@register_dataset("mimic_ext_cxr_qba")
class MIMICExtCXRQBADataset(BaseVQADataset):
    """VQA dataset over MIMIC-CXR studies.

    Args:
        base_image_dir: MIMIC-CXR image root, e.g.
            ``/data/mimic-cxr/2.1.0/files``.
        metadata_dir: Directory containing the QBA parquet metadata files.
        qa_zip_path: Path to ``qa.zip`` for question/answer text extraction.
        image_ext: ``".dcm"`` (default) or ``".jpg"``.
        max_studies: Limit the number of studies processed.  ``None`` reads all.
        transform, cache_dir, output_struct, output_question_type:
            See :class:`BaseVQADataset`.
        harmonized_df, harmonizer, harmonizer_path: Standard harmonizer
            resolution paths; when used, ``metadata_dir`` / ``qa_zip_path``
            are not required.
    """

    _HARMONIZER_CLS = MIMICExtCXRQBAHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        metadata_dir: str = None,
        qa_zip_path: str = None,
        image_ext: str = ".dcm",
        max_studies: int | None = None,
        transform=None,
        cache_dir: str = "./cache",
        output_struct: bool = False,
        output_question_type: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        # When loading from a saved harmonizer, infer base_image_dir.
        if harmonizer_path is not None and base_image_dir is None:
            _h = MIMICExtCXRQBAHarmonizer.load_from_saved(harmonizer_path)
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

        if (
            harmonized_df is None
            and harmonizer is None
            and harmonizer_path is None
        ):
            if metadata_dir is None:
                raise ValueError(
                    "metadata_dir is required when constructing the harmonizer "
                    "from scratch."
                )
            self._harmonizer = MIMICExtCXRQBAHarmonizer(
                metadata_dir=metadata_dir,
                qa_zip_path=qa_zip_path,
                base_image_dir=base_image_dir,
                image_ext=image_ext,
                max_studies=max_studies,
            )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        return self._harmonizer.harmonize()
