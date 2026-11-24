"""VQA dataset for CheXinstruct's MIMIC-CXR-VQA + MIMIC-Diff-VQA subset."""

import os

import torch

from radharmony.harmonizer import ChexInstructVQAHarmonizer
from radharmony.registry import register_dataset
from radharmony.utils.infer import infer_path
from ..base_vqa import BaseVQADataset


@register_dataset("chexinstruct_vqa")
class ChexInstructVQADataset(BaseVQADataset):
    """VQA dataset over CheXinstruct's MIMIC-CXR-VQA + MIMIC-Diff-VQA records, all splits.

    This class mixes the train, val and test splits. It is not for
    benchmarks. Use :class:`ChexInstructVQATrainDataset`,
    :class:`ChexInstructVQAValDataset` and :class:`ChexInstructVQATestDataset`.

    Covers the "Open-Ended VQA", "Close-Ended VQA" and "Difference VQA" tasks.
    Stanford's CheXinstruct derives them from MIMIC-CXR (see
    ``docs/wiki/reference/datasets/vqa-text/chexinstruct_vqa.md`` for the other five
    CheXinstruct VQA sources this does *not* cover, and why).

    ``image_path`` is relative to the image folder of its source (D-04). When
    all sources share one folder (today's case), the VQA base joins it as
    ``base_image_dir``. With several folders in ``source_image_dirs``, this
    class joins the folder of each sample's source.

    Args:
        base_image_dir: One root for the images of every data source. Both
            current sources (MIMIC-CXR-VQA, MIMIC-Diff-VQA) resolve to
            MIMIC-CXR-JPG's ``files/`` directory, for example
            ``/data/mimic-cxr-jpg/2.0.0/files/``. This is the same directory
            as in RadHarmony's ``MIMICCXRJPGDataset``. Ignored when ``source_image_dirs`` is
            given. Also used (if set) as the anchor for ``json_path``
            auto-discovery.
        source_image_dirs: ``{data_source: root_dir}`` mapping for per-source
            image roots, for future CheXinstruct additions that span
            multiple image roots. Takes precedence over ``base_image_dir``.
        json_path: Path to the compiled ``chexinstruct_vqa.json`` produced
            by ``scripts/compile_chexinstruct_vqa.py`` (see the wiki page for the full
            data-acquisition + compile steps). ``infer_path`` searches near
            ``base_image_dir`` when ``None``. The compiled JSON is on a
            separate NAS tree from the MIMIC-CXR-JPG images, unlike VQA-RAD.
            So that search does not find it.
            Pass this explicitly.
        transform: MONAI transform pipeline. Defaults to the standard VQA
            2-D pipeline (loads ``img``, text fields pass through unchanged).
        cache_dir: Root directory for MONAI PersistentDataset cache.
        output_struct: Include ``answer_struct`` in each sample (the
            ``{"yes": bool, "no": bool}`` dict, present on close-ended rows).
        output_question_type: Include ``question_type`` in each sample
            (``"Open-Ended VQA"`` / ``"Close-Ended VQA"`` / ``"Difference VQA"``).
        question_types: Optional subset of
            ``ChexInstructVQAHarmonizer.QUESTION_TYPES`` to keep (for example
            ``["Open-Ended VQA"]``, or any combination of the three). ``None``
            keeps all three task types. It also filters a given table
            (``harmonizer_path``, ``harmonizer`` or ``harmonized_df``, D-02).
        data_sources: Optional subset of
            ``ChexInstructVQAHarmonizer.DATA_SOURCES`` to keep (for example
            ``["MIMIC-CXR-VQA"]`` to drop MIMIC-Diff-VQA rows entirely).
            ``None`` keeps both. It also filters a given table (D-02).
        harmonized_df, harmonizer, harmonizer_path: Standard harmonizer
            resolution paths. When used, ``json_path`` is not required.
    """

    _HARMONIZER_CLS = ChexInstructVQAHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        source_image_dirs: dict = None,
        json_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_struct: bool = False,
        output_question_type: bool = False,
        question_types: list[str] = None,
        data_sources: list[str] = None,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if (
            base_image_dir is None
            and source_image_dirs is None
            and harmonized_df is None
            and harmonizer is None
            and harmonizer_path is None
        ):
            raise ValueError(
                "base_image_dir or source_image_dirs is required when "
                "harmonizer_path / harmonizer / harmonized_df are not provided."
            )

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            self._json_path = infer_path(
                base_image_dir or "", "chexinstruct_vqa.json", user_path=json_path or "",
                required=True, arg="json_path",
            ) or json_path
        else:
            self._json_path = json_path

        if base_image_dir is None and source_image_dirs is None and (harmonizer is not None or harmonizer_path):
            _h = harmonizer if harmonizer is not None else self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_image_dir, source_image_dirs = _h.base_image_dir, _h.source_image_dirs
        roots = set(source_image_dirs.values()) if source_image_dirs else ({base_image_dir} - {None})
        # One shared folder: the VQA base joins it. Several: _make_dataset joins each one (D-04, Q-11).
        self._source_image_dirs = dict(source_image_dirs) if len(roots) > 1 else None
        self._img_source: dict = {}
        super().__init__(
            base_image_dir=roots.pop() if len(roots) == 1 else None,
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
        self._harmonizer = ChexInstructVQAHarmonizer(
            json_path=self._json_path,
            base_image_dir=base_image_dir,
            source_image_dirs=source_image_dirs,
            question_types=question_types,
            data_sources=data_sources,
        )

    def _get_harmonized_df(self):
        preset = self._try_resolve_preset_harmonized()
        df = self._harmonizer.filter_table(preset) if preset is not None else self._harmonizer.harmonize()
        if self.base_image_dir:
            # A table that an older version saved has absolute paths. Make them relative (D-04).
            df = df.assign(image_path=[os.path.relpath(p, self.base_image_dir) if os.path.isabs(p) else p
                                       for p in df["image_path"].astype(str)])
        if self._source_image_dirs:
            # The mixed class repeats an id in each split, so the key is the id and the image.
            keys = zip(df["question_id"].astype(str), df["image_path"].astype(str), strict=True)
            self._img_source = dict(zip(keys, df["data_source"], strict=True))
        return df

    def verify_images(self, drop_missing: bool = True):
        """Check that every image exists, and drop the missing rows if ``drop_missing``.

        With several ``source_image_dirs``, each ``image_path`` is relative to the folder of its
        ``data_source`` (D-04). A source with no folder counts as missing. With one folder, the VQA base
        method does the check.

        Returns:
            DataFrame of the rows whose image file was not found.
        """
        if not self._source_image_dirs:
            return super().verify_images(drop_missing)
        df = self._get_harmonized_df()
        dirs = self._source_image_dirs
        found = [s in dirs and os.path.isfile(os.path.join(dirs[s], p))
                 for s, p in zip(df["data_source"], df["image_path"].astype(str), strict=True)]
        missing_df = df[[not f for f in found]].copy()
        if not missing_df.empty and drop_missing:
            kept = df[found].reset_index(drop=True)
            self.set_harmonized_df(kept)
            print(f"Dropped {len(missing_df)} rows with missing images ({len(kept)} remaining).")
        elif missing_df.empty:
            print("All image files exist.")
        else:
            print(f"Found {len(missing_df)} rows with missing images (not dropped).")
        return missing_df

    def _make_dataset(self, dicts: list, cache_subdir: str, transform=None):
        if self._source_image_dirs:
            # Several image folders: join the folder of the source of each sample (D-04).
            dicts = [dict(d, img=os.path.join(self._source_image_dirs[self._img_source[(d["question_id"], d["img"])]],
                                              d["img"]))
                     for d in dicts]
        return super()._make_dataset(dicts, cache_subdir, transform)
