"""MONAI dataset for the MS-CXR-T temporal image pairs benchmark.

1,045 longitudinal chest X-ray pairs from MIMIC-CXR-JPG with progression
labels (stable / improving / worsening) for up to 5 pathology findings.

``base_image_dir`` is the ``files/`` folder of MIMIC-CXR-JPG 2.0.0, the same
folder as for MS-CXR (D-04).  Each sample contains both ``"img"`` (current visit) and
``"previous_img"`` (prior visit) tensors when ``output_previous=True``.
"""

import collections.abc
import os

import pandas as pd
import torch

from radharmony.harmonizer.ms_cxr_t import MSCXRTHarmonizer
from radharmony.registry import register_dataset
from radharmony.dataset.base import BaseRadiologicalDataset
from radharmony.dataset.transforms import RadiologyTransform2D
from radharmony.utils.infer import infer_path


class _WithPrevious(torch.utils.data.Dataset):
    """Add the prior-visit image to each sample under ``"previous_img"`` (V-D-31).

    Args:
        current: dataset of the current images. Its ``data`` and ``transform`` stay visible.
        previous: dataset of the prior images, with the same transform and cache.
    """

    def __init__(self, current, previous):
        self.current, self.previous = current, previous
        self.data, self.transform = current.data, current.transform

    def __len__(self):
        return len(self.current)

    def __getitem__(self, index):
        # Slice and list access give a Subset, as in a MONAI Dataset.
        if isinstance(index, slice):
            return torch.utils.data.Subset(self, range(len(self))[index])
        if isinstance(index, collections.abc.Sequence):
            return torch.utils.data.Subset(self, index)
        sample = dict(self.current[index])
        # Both datasets share one transform. The current dataset seeds it once in each
        # DataLoader worker. A second seed would repeat the random draw of img.
        self.previous._worker_seed = getattr(self.current, "_worker_seed", None)
        sample["previous_img"] = self.previous[index]["img"]
        return sample


@register_dataset("ms_cxr_t")
class MSCXRTDataset(BaseRadiologicalDataset):
    """PyTorch/MONAI dataset for the MS-CXR-T temporal progression benchmark.

    Args:
        base_image_dir: The ``files/`` folder of MIMIC-CXR-JPG 2.0.0.
        csv_path: Path to ``MS_CXR_T_temporal_image_classification_v1.0.0.csv``. Found in
            ``base_image_dir`` or its parent when ``None``. It is usually in
            its own folder, so pass it.
        transform: MONAI Compose transform.  Defaults to standard 2-D 224 px.
        cache_dir: MONAI PersistentDataset cache directory.
        output_previous: Also give the prior-visit image under ``"previous_img"``. It goes
            through the same transform as ``img`` in a second call, so a random step draws
            again for it (D-11).
        dtype: Dtype of ``img``. The labels stay float32. Default ``torch.bfloat16``.
    """

    SUPPORTED_OUTPUTS: frozenset = frozenset({"previous_img"})
    LABEL_COLS = []
    _HARMONIZER_CLS = MSCXRTHarmonizer

    def __init__(
        self,
        base_image_dir: str = None,
        csv_path: str = None,
        transform=None,
        cache_dir: str = "./cache",
        output_cls: bool = False,
        output_mask: bool = False,
        output_report: bool = False,
        output_reg: bool = False,
        output_bbox: bool = False,
        output_previous: bool = False,
        harmonized_df=None,
        harmonizer=None,
        harmonizer_path: str = None,
        dtype=torch.bfloat16,
    ):
        if base_image_dir:
            base_image_dir = os.path.expanduser(base_image_dir)
        if csv_path:
            csv_path = os.path.expanduser(csv_path)

        if harmonizer_path is not None and base_image_dir is None:
            _h = self._HARMONIZER_CLS.load_from_saved(harmonizer_path)
            base_image_dir = getattr(_h, "mimic_base_dir", None)

        if (
            base_image_dir is None
            and harmonizer_path is None
            and harmonized_df is None
            and harmonizer is None
        ):
            raise ValueError(
                "base_image_dir (the MIMIC-CXR-JPG 2.0.0 files/ folder) is required when "
                "harmonizer_path is not provided."
            )

        self._output_previous = output_previous

        if transform is None:
            transform = RadiologyTransform2D(
                img_size=224,
                output_keys={"img"},
                dtype=dtype,
            ).get_transform()

        if harmonizer_path is None and harmonized_df is None and harmonizer is None:
            # The CSV is in its own folder, not near the MIMIC files/ folder, so the
            # user usually passes it. Without it, the error names the file (D-35).
            csv_path = infer_path(
                base_image_dir, "MS_CXR_T_temporal_image_classification_v1.0.0.csv", user_path=csv_path or "",
                required=True, arg="csv_path",
            )
        self._csv_path = csv_path
        super().__init__(
            base_image_dir=base_image_dir,
            transform=transform,
            cache_dir=cache_dir,
            output_cls=output_cls,
            output_mask=output_mask,
            output_report=output_report,
            output_reg=output_reg,
            output_bbox=output_bbox,
            harmonized_df=harmonized_df,
            harmonizer=harmonizer,
            harmonizer_path=harmonizer_path,
        )

    @property
    def _cols(self) -> dict:
        cols = super()._cols
        if self._output_previous:
            cols["previous_img"] = "previous_image_path"  # a row with no prior image is dropped (D-11)
        return cols

    def _make_dataset(self, dicts: list, cache_subdir: str, transform=None):
        current = super()._make_dataset(dicts, cache_subdir, transform)
        if not self._output_previous:
            return current
        if dicts and "previous_img" not in dicts[0]:
            raise ValueError("output_previous=True needs a previous_image_path column in the table (V-D-31).")
        root = self.base_image_dir
        prev = [{"img": os.path.join(root, d["previous_img"]) if root else d["previous_img"]} for d in dicts]
        return _WithPrevious(current, super()._make_dataset(prev, cache_subdir, transform))

    def _get_harmonized_df(self) -> pd.DataFrame:
        preset = self._try_resolve_preset_harmonized()
        if preset is not None:
            return preset
        h = MSCXRTHarmonizer(
            csv_path=self._csv_path,
            mimic_base_dir=self.base_image_dir,
        )
        return h.harmonize()
