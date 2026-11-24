"""Harmonizer for MedPMC: 11M medical image-text pairs from PubMed Central.

Source: https://huggingface.co/datasets/Yale-BIDS-Chen/medpmc-11m-dataset_jun24_baseline
Paper: Kim et al., MedPMC, arXiv:2607.07673. The license of the data is CC BY-NC-SA-4.0
(non-commercial, share-alike, from the Hugging Face card). The CC BY 4.0 of the arXiv
page is the license of the paper, not of the data. Each record has its own source and
license metadata. Check them before you share data.

MedPMC is a caption corpus, so it harmonizes to the VQA schema as ROCO does.
``question`` is the empty string and ``answer`` is the caption.

Layout (after ``huggingface_hub.snapshot_download``)::

    <base_dir>/
      webdataset/
        metadata/medpmc_index.parquet     # 11,123,087 rows
        multipanel/train-000000.tar ...   # 542 shards
        singlepanel/train-000000.tar ...  #  73 shards
        subfigure/train-000000.tar ...    # 100 shards
      tags/medpmc_tags.parquet            # built by us, see scripts/build_medpmc_tags.py

The tag table is not part of the MedPMC release. The release has no
specialty or modality field. ``scripts/build_medpmc_tags.py`` makes one from
the caption text, and ``tag_method`` on every row says how. Without the tag
table the harmonizer warns, cannot filter by specialty, and every ``answer``
is None (the caption comes from the tag table).

The images are inside webdataset tars. ``image_path`` is
``<source_type>/<key><ext>``, relative to ``base_image_dir``. ``shard`` and
``key`` let ``scripts/materialize_medpmc_subset.py`` extract a subset. Call
``verify_images`` after the extraction, not before.

Benchmark leak: PMC figures include images of MIMIC-CXR, CheXpert, VinDr and
other benchmarks. A model pretrained on MedPMC and tested on them may have
seen the test images. Treat such results as suspect unless the overlap is
removed.

Duplicates: a subfigure and its multipanel parent are both rows. Mixing the
two subsets counts the same pixels twice. Group by ``parent_image_id`` first.

"Radiology" is not "chest X-ray". CT and MRI dominate the radiology tag. Use
``modalities=["xray"]`` for X-ray.

Modality source: ``modalities=[...]`` uses the CLIP label by default, and the
caption tagger where CLIP is null. In a clinician check of 151 tagger
conflicts on 2026-08-22, CLIP was right 72.8% of the time and the caption
lexicon 10.6%. Use ``modality_source="caption"`` for the old behavior or
``"either"`` for the union of both.

Harmonized columns: patient_id, study_id, question_id, image_path, question,
answer. MedPMC has no patients, so ``patient_id`` is the PMC article id. It is
the nearest choice for a split without leaks, because panels of one figure and
figures of one article are not independent.
"""
from __future__ import annotations

import os
import warnings

import pandas as pd

from ..base_vqa import BaseVQAHarmonizer
from .tagging import RADIOLOGY_MODALITIES, SPECIALTIES

_SUBSETS = ("multipanel", "singlepanel", "subfigure")


class MedPMCHarmonizer(BaseVQAHarmonizer):
    """Harmonize the MedPMC webdataset index into the RadHarmony VQA schema.

    ``include_multipanel=False`` drops the multipanel rows from a specialty
    filter (``radiology_only`` or ``specialties``), because their tag comes from
    the caption of the whole figure. Without a specialty filter, it does nothing.
    """

    EXTRA_OUTPUT_COLS = ["source_type", "pmcid", "image_id", "parent_image_id",
                         "subfigure_index", "figure_label", "shard", "key",
                         "specialty", "modality", "tag_confidence", "tag_source", "tag_method",
                         "tag_scope", "contains_specialty",
                         "clip_specialty", "clip_modality", "clip_prob"]

    def __init__(
        self,
        base_dir: str = None,
        base_image_dir: str = None,
        subsets: list[str] | None = None,
        radiology_only: bool = False,
        specialties: list[str] | None = None,
        modalities: list[str] | None = None,
        min_tag_confidence: str | None = None,
        drop_untagged: bool = False,
        require_tagger_agreement: bool = False,
        min_clip_prob: float = None,
        modality_source: str = "clip",
        clip_tags_path: str = None,
        include_multipanel: bool = True,
        tags_path: str | None = None,
    ):
        super().__init__(base_image_dir=base_image_dir)
        self.base_dir = os.path.expanduser(base_dir) if base_dir else None
        self.subsets = list(subsets) if subsets else list(_SUBSETS)
        self.radiology_only = radiology_only
        self.specialties = list(specialties) if specialties else None
        self.modalities = list(modalities) if modalities else None
        if modality_source not in ("clip", "caption", "either"):
            raise ValueError("modality_source must be clip, caption or either")
        self.modality_source = modality_source
        self.min_tag_confidence = min_tag_confidence
        self.drop_untagged = drop_untagged
        self.require_tagger_agreement = require_tagger_agreement
        self.min_clip_prob = min_clip_prob
        self.clip_tags_path = clip_tags_path
        self.include_multipanel = include_multipanel
        self.tags_path = tags_path

        unknown = set(self.subsets) - set(_SUBSETS)
        if unknown:
            raise ValueError(f"unknown MedPMC subsets: {sorted(unknown)}; valid: {_SUBSETS}")
        if self.specialties:
            bad = set(self.specialties) - set(SPECIALTIES)
            if bad:
                raise ValueError(f"unknown specialties: {sorted(bad)}; valid: {SPECIALTIES}")
        if self.modalities:
            bad = set(self.modalities) - set(RADIOLOGY_MODALITIES)
            if bad:
                raise ValueError(f"unknown modalities: {sorted(bad)}; valid: {RADIOLOGY_MODALITIES}")
        if self.min_tag_confidence not in (None, "low", "high"):
            raise ValueError("min_tag_confidence must be None, 'low' or 'high'")

    def _harmonizer_init_snapshot(self) -> dict:
        snap = super()._harmonizer_init_snapshot()  # keeps base_image_dir (D-08)
        for k, default in (("base_dir", None), ("subsets", list(_SUBSETS)),
                           ("radiology_only", False), ("specialties", None),
                           ("modalities", None), ("min_tag_confidence", None),
                           ("drop_untagged", False), ("require_tagger_agreement", False),
                           ("min_clip_prob", None), ("modality_source", "clip"),
                           ("clip_tags_path", None), ("include_multipanel", True),
                           ("tags_path", None)):
            v = getattr(self, k)
            if v != default:
                snap[k] = v
        return snap

    # ------------------------------------------------------------------ helpers
    def _default_tags_path(self) -> str:
        return os.path.join(self.base_dir, "tags", "medpmc_tags.parquet")

    def _load_tags(self) -> pd.DataFrame | None:
        p = self.tags_path or self._default_tags_path()
        if not os.path.isfile(p):
            return None
        t = pd.read_parquet(p)
        need = {"image_id", "specialty", "modality", "tag_confidence", "tag_method"}
        missing = need - set(t.columns)
        if missing:
            raise ValueError(f"tag table {p} missing columns: {sorted(missing)}")
        return t

    # ----------------------------------------------------------------- harmonize
    def _effective_modality(self, df):
        """Modality actually used for filtering, per ``modality_source``.

        clip     -> the CLIP label, or the caption tagger where CLIP is null
        caption  -> the caption lexicon only (the default before 2026-08-22)
        either   -> a match if either tagger says it (best recall, worst precision)
        """
        cap = df["modality"] if "modality" in df.columns else None
        clip = df["clip_modality"] if "clip_modality" in df.columns else None
        if self.modality_source == "caption" or clip is None:
            return cap
        if self.modality_source == "clip":
            return clip.where(clip.notna(), cap)
        return cap  # "either" is handled by the caller via a union mask

    def _read_source(self) -> pd.DataFrame:
        """Read the index, join the tag tables, apply the filters and return the table."""
        if self.base_dir is None:
            raise ValueError("MedPMCHarmonizer requires base_dir")

        idx_path = os.path.join(self.base_dir, "webdataset", "metadata",
                                "medpmc_index.parquet")
        if not os.path.isfile(idx_path):
            raise FileNotFoundError(
                f"MedPMC index not found at {idx_path}. Download the corpus first "
                f"(huggingface_hub.snapshot_download of "
                f"Yale-BIDS-Chen/medpmc-11m-dataset_jun24_baseline)."
            )
        df = pd.read_parquet(idx_path)
        df = df[df["source_type"].isin(self.subsets)].copy()

        # Agreement compares the caption tag with the CLIP tag, so it needs the caption tags too.
        wants_tag_filter = bool(self.radiology_only or self.specialties
                                or self.modalities or self.min_tag_confidence
                                or self.drop_untagged or self.require_tagger_agreement)
        tags = self._load_tags()
        if tags is None:
            if wants_tag_filter:
                raise FileNotFoundError(
                    "A specialty/modality filter was requested but no tag table was "
                    f"found at {self.tags_path or self._default_tags_path()}. MedPMC "
                    "ships no specialty field; build one with "
                    "scripts/build_medpmc_tags.py first."
                )
            warnings.warn(
                "MedPMC: no tag table at "
                f"{self.tags_path or self._default_tags_path()}, so the caption "
                "(answer) of every row is None. Build the table with "
                "scripts/build_medpmc_tags.py.",
                UserWarning,
                stacklevel=3,  # the caller of harmonize()
            )
            for c in ("caption", "specialty", "modality", "tag_confidence",
                      "tag_source", "tag_method"):
                df[c] = None
        else:
            # tag_scope and contains_specialty feed the multi-panel filter below (V-H1-21).
            keep = [c for c in ("image_id", "caption", "specialty", "modality",
                                "tag_confidence", "tag_source", "tag_method",
                                "tag_scope", "contains_specialty")
                    if c in tags.columns]
            df = df.merge(tags[keep], on="image_id", how="left")

        # Optional second opinion from the zero-shot CLIP tagger.
        # Neither tagger works alone. In a check of ten conflict images the
        # caption lexicon was right on 5 of 7 and CLIP on 3 of 7, on different
        # images. The lexicon called a clinical photo radiology, and CLIP called a
        # breast MRI non-medical. Agreement lets each tagger veto the other.
        # CLIP was right both times it was confident, and wrong or unsure at
        # p <= 0.72, so min_clip_prob is a second control.
        # This filter favors precision and keeps about 649k of the 1.76M
        # images that the lexicon calls radiology. Do not use it to estimate prevalence.
        # The CLIP table must also load when it decides the modality. Else
        # modality_source="clip" or "either" falls back to the caption tagger
        # and does nothing (all three sources gave 221,896 for xray).
        _needs_clip = (self.require_tagger_agreement
                       or self.min_clip_prob is not None
                       or (self.modalities and self.modality_source in ("clip", "either")))
        if _needs_clip:
            cp = self.clip_tags_path or os.path.join(self.base_dir, "tags",
                                                     "medpmc_tags_clip.parquet")
            if not os.path.isfile(cp):
                raise FileNotFoundError(
                    f"tagger agreement requested but no CLIP tag table at {cp}. "
                    f"Build it with scripts/build_medpmc_tags_clip.py."
                )
            ct = pd.read_parquet(cp, columns=["key", "specialty", "modality", "clip_prob"])
            ct = ct.rename(columns={"specialty": "clip_specialty",
                                    "modality": "clip_modality"})
            df = df.merge(ct, on="key", how="left")

        return self.filter_table(df)

    def filter_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the subset, tagger, specialty, modality and confidence filters (D-02).

        It works on the source rows after the tag joins and on a harmonized table, which have the same
        column names.

        Args:
            df: The source rows or a harmonized table.

        Returns:
            The kept rows, with a new index.

        Raises:
            ValueError: If a filter needs a column that the table does not have.
        """
        def need(*cols):
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(f"MedPMC: the table has no column {missing}, which a filter needs. "
                                 "Harmonize again with this filter, or remove it.")

        if set(self.subsets) != set(_SUBSETS):
            need("source_type")
            df = df[df["source_type"].isin(self.subsets)]
        if self.require_tagger_agreement:
            need("specialty", "modality", "clip_specialty", "clip_modality")
            df = df[(df["specialty"] == df["clip_specialty"])
                    & (df["modality"].isna() | df["clip_modality"].isna()
                       | (df["modality"] == df["clip_modality"]))]
        if self.min_clip_prob is not None:
            need("clip_prob")
            df = df[df["clip_prob"] >= self.min_clip_prob]

        # radiology_only is a short form of specialties=["radiology"].
        #
        # SCOPE. Multi-panel figures (3.1M, 28% of the corpus) cannot carry one
        # specialty for each panel. Their tag comes from the caption of the whole
        # figure, so it says what the figure contains. include_multipanel=True keeps
        # them in a specialty filter. False drops them, for a panel-level set
        # (V-H1-47). An optional contains_specialty column also keeps a multipanel
        # row whose caption names the specialty.
        specialties = self.specialties or (["radiology"] if self.radiology_only else None)
        if specialties:
            need("specialty", "source_type")
            sel = df["specialty"].isin(specialties)
            mp = df["source_type"].eq("multipanel")
            if not self.include_multipanel:
                sel = sel & ~mp
            elif "contains_specialty" in df.columns:
                sel = sel | (mp & df["contains_specialty"].fillna("").str.contains(
                    "|".join(specialties), case=False, regex=True))
            df = df[sel]
        if self.modalities:
            # harmonize() joins the CLIP table for "clip" and "either", so a given table needs it too.
            need("modality", *(("clip_modality",) if self.modality_source in ("clip", "either") else ()))
            # The default tagger changed to CLIP on 2026-08-22. Of 151 conflicts that a
            # clinician judged as radiology, CLIP was right on 72.8% and the
            # caption lexicon on 10.6%. The old default picked the worse source
            # on about 386K images. Use modality_source="caption" for the old
            # behavior.
            if self.modality_source == "either" and "clip_modality" in df.columns:
                df = df[df["modality"].isin(self.modalities)
                        | df["clip_modality"].isin(self.modalities)]
            else:
                df = df[self._effective_modality(df).isin(self.modalities)]
        if self.min_tag_confidence:
            need("tag_confidence")
        if self.min_tag_confidence == "high":
            df = df[df["tag_confidence"] == "high"]
        elif self.min_tag_confidence == "low":
            df = df[df["tag_confidence"].isin(("low", "high"))]
        if self.drop_untagged:
            need("specialty")
            df = df[df["specialty"].notna() & (df["specialty"] != "unknown")]

        return df.reset_index(drop=True)

    def _build_vqa_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        ext = df["image_ext"].fillna(".jpg")
        out = pd.DataFrame({
            # No patients exist in a literature corpus. The PMC article id is the
            # correct grouping unit for leakage-aware splits: panels of one figure
            # and figures of one article are not independent samples.
            "patient_id": df["pmcid"],
            "study_id": df["parent_image_id"].fillna(df["image_id"]),
            "question_id": df["image_id"],
            "image_path": df["source_type"] + "/" + df["key"] + ext,
            "question": "",
            "answer": df["caption"] if "caption" in df.columns else None,
        })
        for c in self.EXTRA_OUTPUT_COLS:
            if c in df.columns:
                out[c] = df[c]
        return out
