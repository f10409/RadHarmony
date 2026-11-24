"""MONAI dataset for ROCOv2 (Radiology Objects in COntext, version 2)."""

import os

import torch

from radharmony.harmonizer.rocov2 import ROCOv2Harmonizer
from radharmony.registry import register_dataset

from ..base_vqa import BaseVQADataset


@register_dataset("rocov2")
class ROCOv2Dataset(BaseVQADataset):
    """Captioning dataset built on ROCOv2: 79,793 radiology figure-caption pairs in the Hugging Face files.

    Successor to ROCO, from the PMC Open Access subset. ``question`` is always the
    empty string; ``answer`` is the caption.

    **Per-image modality labels.** Unlike ROCO's binary radiology/non-radiology
    directory split, ROCOv2 ships UMLS concepts per image, and the Zenodo release
    adds human-curated ``<split>_concepts_manual.csv``. Those are resolved to a
    single ``modality`` (ct / mri / xray / ultrasound / angiography / pet_spect /
    mammography) using the canonical names in ``cui_mapping.csv``. Images whose
    concepts name several modalities get ``modality=None`` rather than an arbitrary
    pick: an image really can be both CT and PET.

    **Radiology only.** This corpus has no non-radiology negatives. It can
    measure modality accuracy and radiology recall, but never a false-positive
    rate against other specialties. Do not use it to claim a tagger "doesn't
    over-call radiology".

    Args:
        base_dir: ROCOv2 root containing ``hf/data/`` plus the Zenodo CSVs.
        base_image_dir: Where materialised images live. With ``harmonizer_path``,
            ``None`` gives the ``base_image_dir`` of the saved harmonizer. Else it gives
            ``<base_dir>/images`` (see ``scripts/materialize_rocov2.py``).
        splits: Any of ``train`` / ``validation`` / ``test``. Defaults to all three.
        modalities: Restrict to these modalities.
        require_modality: Drop images with no single unambiguous modality.
        manual_concepts_only: Prefer the curated Zenodo concepts over the
            auto-extracted CUIs in the HF parquet (default True).
        transform, cache_dir, output_struct, output_question_type:
            See :class:`~radharmony.dataset.base_vqa.BaseVQADataset`.
        harmonized_df, harmonizer, harmonizer_path: Standard harmonizer
            resolution paths. When used, ``base_dir`` is not required. ``splits``,
            ``modalities`` and ``require_modality`` also filter this table (D-02).
            ``manual_concepts_only`` does not change it.

    Example: CT and MRI from the test split only::

        ds = ROCOv2Dataset(base_dir="/data/ROCOv2",
                           splits=["test"], modalities=["ct", "mri"],
                           require_modality=True)
    """

    _HARMONIZER_CLS = ROCOv2Harmonizer
    IMG_CHANNELS = 3  # color figures (D-42)

    def __init__(
        self,
        base_dir: str = None,
        base_image_dir: str = None,
        splits: list[str] | None = None,
        modalities: list[str] | None = None,
        require_modality: bool = False,
        manual_concepts_only: bool = True,
        transform=None,
        cache_dir: str = "./cache",
        output_struct: bool = False,
        output_question_type: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if harmonizer_path is not None and (base_dir is None or base_image_dir is None):
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_dir = base_dir or _h.base_dir
            base_image_dir = base_image_dir or _h.base_image_dir

        if (base_dir is None and harmonized_df is None
                and harmonizer is None and harmonizer_path is None):
            raise ValueError(
                "base_dir (ROCOv2 root containing hf/data/) is required when "
                "harmonizer_path / harmonizer / harmonized_df are not provided."
            )

        if base_image_dir is None and base_dir is not None:
            base_image_dir = os.path.join(os.path.expanduser(base_dir), "images")

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

        self._harmonizer = self._HARMONIZER_CLS(
            base_dir=base_dir,
            base_image_dir=base_image_dir,
            splits=splits,
            modalities=modalities,
            require_modality=require_modality,
            manual_concepts_only=manual_concepts_only,
        )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            # A given table gets the same filters (D-02).
            return self._harmonizer.filter_table(preset)
        return self._harmonizer.harmonize()
