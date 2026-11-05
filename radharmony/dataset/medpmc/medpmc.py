"""MONAI dataset for MedPMC: 11M medical image-text pairs curated from PubMed Central."""

import os

import torch

from radharmony.harmonizer.medpmc import MedPMCHarmonizer
from radharmony.registry import register_dataset

from ..base_vqa import BaseVQADataset


@register_dataset("medpmc")
class MedPMCDataset(BaseVQADataset):
    """Captioning dataset built on MedPMC (arXiv:2607.07673).

    Each sample pairs a medical figure, single panel, or extracted subfigure with
    its PubMed Central caption. ``question`` is always the empty string (pure
    captioning); ``answer`` is the caption.

    **Specialty filtering.** MedPMC ships no specialty or modality field. The
    breakdown in the paper came from a manual audit of 2,906 sampled images,
    not from per-image labels. RadHarmony derives its own tag table from
    caption text (``scripts/build_medpmc_tags.py``). Every row carries
    ``tag_method``, so these tags are never mistaken for author-provided
    ground truth. Requesting any specialty filter without a tag table raises
    rather than silently returning everything.

    **Images live inside webdataset tars.** ``image_path`` is written relative to
    ``base_image_dir`` as ``<source_type>/<key><ext>``. Extract the subset you
    actually want with ``scripts/materialize_medpmc_subset.py`` before iterating.
    Extracting all 11M files is a bad idea on network storage.

    Args:
        base_dir: Root of the MedPMC download, the directory containing
            ``webdataset/`` (and ``tags/`` once built).
        base_image_dir: Directory holding materialised images. With ``harmonizer_path``,
            ``None`` gives the ``base_image_dir`` of the saved harmonizer. Else it gives
            ``<base_dir>/images``.
        subsets: Which of ``multipanel`` / ``singlepanel`` / ``subfigure`` to
            load. Defaults to all three.
        radiology_only: Convenience flag equivalent to
            ``specialties=["radiology"]``. Requires a tag table.
        specialties: Restrict to these specialties, for example ``["radiology",
            "pathology"]``. See ``radharmony.harmonizer.medpmc.SPECIALTIES``.
        modalities: Restrict to these radiology sub-modalities, for example
            ``["ct", "mri"]``. See
            ``radharmony.harmonizer.medpmc.RADIOLOGY_MODALITIES``.
        min_tag_confidence: ``"high"`` keeps only rows tagged from a specific
            term; ``"low"`` also keeps generic matches. ``None`` keeps everything.
        drop_untagged: Drop rows whose specialty could not be determined.
        require_tagger_agreement: Keep only images where the caption lexicon and the
            zero-shot pixel tagger agree on specialty (and on modality where both
            assign one). This is a precision filter. See the harmonizer docstring for
            why neither tagger is trustworthy alone.
        min_clip_prob: Minimum zero-shot confidence. The pixel tagger was correct
            both times it exceeded 0.97 on expert adjudication. It was wrong or
            undecidable at 0.72 and below. A setting of 0.9 is reasonable.
        modality_source: The tagger that ``modalities`` reads: ``"clip"`` (default),
            ``"caption"`` or ``"either"``. See :class:`MedPMCHarmonizer`.
        include_multipanel: ``False`` drops the multipanel rows from a specialty
            filter. See :class:`MedPMCHarmonizer`.
        clip_tags_path: Explicit path to the pixel-tagger table.
        tags_path: Explicit path to a tag table, overriding
            ``<base_dir>/tags/medpmc_tags.parquet``.
        transform, cache_dir, output_struct, output_question_type:
            See :class:`~radharmony.dataset.base_vqa.BaseVQADataset`.
        harmonized_df, harmonizer, harmonizer_path: Standard harmonizer
            resolution paths. When used, ``base_dir`` is not required. The
            filter arguments also filter this table (D-02).

    Example: radiology only, CT and MRI, confident tags only::

        ds = MedPMCDataset(
            base_dir="/data/MedPMC",
            radiology_only=True,
            modalities=["ct", "mri"],
            min_tag_confidence="high",
        )
    """

    _HARMONIZER_CLS = MedPMCHarmonizer
    IMG_CHANNELS = 3  # color figures (D-42)

    def __init__(
        self,
        base_dir: str = None,
        base_image_dir: str = None,
        subsets: list[str] | None = None,
        radiology_only: bool = False,
        specialties: list[str] | None = None,
        modalities: list[str] | None = None,
        min_tag_confidence: str = None,
        drop_untagged: bool = False,
        require_tagger_agreement: bool = False,
        min_clip_prob: float = None,
        modality_source: str = "clip",
        include_multipanel: bool = True,
        clip_tags_path: str = None,
        tags_path: str = None,
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

        if (
            base_dir is None
            and harmonized_df is None
            and harmonizer is None
            and harmonizer_path is None
        ):
            raise ValueError(
                "base_dir (MedPMC root containing webdataset/) is required "
                "when harmonizer_path / harmonizer / harmonized_df are not provided."
            )

        # Images are materialised out of the webdataset tars into <base_dir>/images
        # by scripts/materialize_medpmc_subset.py, so that is the image root unless
        # the caller points somewhere else.
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

        # With a given table, the harmonizer only filters it (D-02).
        self._harmonizer = MedPMCHarmonizer(
            base_dir=base_dir,
            base_image_dir=base_image_dir,
            subsets=subsets,
            radiology_only=radiology_only,
            specialties=specialties,
            modalities=modalities,
            min_tag_confidence=min_tag_confidence,
            drop_untagged=drop_untagged,
            require_tagger_agreement=require_tagger_agreement,
            min_clip_prob=min_clip_prob,
            modality_source=modality_source,
            include_multipanel=include_multipanel,
            clip_tags_path=clip_tags_path,
            tags_path=tags_path,
        )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return self._harmonizer.filter_table(preset)
        return self._harmonizer.harmonize()
