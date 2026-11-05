"""VQA datasets for the official CheXinstruct VQA train, val and test splits."""

from radharmony.harmonizer import (
    ChexInstructVQATestHarmonizer,
    ChexInstructVQATrainHarmonizer,
    ChexInstructVQAValHarmonizer,
)
from radharmony.registry import register_dataset
from .chexinstruct_vqa import ChexInstructVQADataset


class _ChexInstructVQASplitDataset(ChexInstructVQADataset):
    """CheXinstruct VQA dataset for one official split (see the subclasses).

    The arguments are the same as :class:`ChexInstructVQADataset`. The
    authors did not split by patient. 2,007 patients are in more than one
    split (see the wiki page).
    """

    def __init__(self, base_image_dir: str = None, source_image_dirs: dict = None,
                 json_path: str = None, *, question_types: list[str] = None,
                 data_sources: list[str] = None, **kwargs):
        super().__init__(base_image_dir=base_image_dir, source_image_dirs=source_image_dirs,
                         json_path=json_path, question_types=question_types,
                         data_sources=data_sources, **kwargs)
        if hasattr(self, "_harmonizer"):
            self._harmonizer = self._HARMONIZER_CLS(
                json_path=self._json_path,
                base_image_dir=base_image_dir,
                source_image_dirs=source_image_dirs,
                question_types=question_types,
                data_sources=data_sources,
            )


@register_dataset("chexinstruct_vqa_train")
class ChexInstructVQATrainDataset(_ChexInstructVQASplitDataset):
    """CheXinstruct VQA official train split (580,166 questions)."""

    _HARMONIZER_CLS = ChexInstructVQATrainHarmonizer


@register_dataset("chexinstruct_vqa_val")
class ChexInstructVQAValDataset(_ChexInstructVQASplitDataset):
    """CheXinstruct VQA official val split (99,996 questions)."""

    _HARMONIZER_CLS = ChexInstructVQAValHarmonizer


@register_dataset("chexinstruct_vqa_test")
class ChexInstructVQATestDataset(_ChexInstructVQASplitDataset):
    """CheXinstruct VQA official test split (21,296 questions)."""

    _HARMONIZER_CLS = ChexInstructVQATestHarmonizer
