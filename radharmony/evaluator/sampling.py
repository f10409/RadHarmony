"""Row, patient-id and random helpers that the evaluators share (not base code)."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit

from ..utils.data_utils import kept_rows


def dataset_rows(ds):
    """Return the table rows that the dataset gives as samples, in sample order.

    It uses the row filter of ``get_data_dict``, so the ids match the samples
    (V-E-9). For example, ``output_mask=True`` drops a row with an empty
    ``mask_path``.
    """
    return kept_rows(ds._get_harmonized_df(), "image_path", getattr(ds, "_cols", None))


# The image file endings that image_ids cuts. A 3-D study folder, such as a dotted UID, has none.
_IMAGE_ENDINGS = (".nii.gz", ".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".nii", ".mha", ".nrrd",
                  ".tif", ".tiff", ".bmp", ".npy")


def _cut_image_ending(path: str) -> str:
    """Return ``path`` with no image file ending. ``1.2.826.0.1`` stays whole, ``a.nii.gz`` gives ``a``."""
    low = path.lower()
    for end in _IMAGE_ENDINGS:
        if low.endswith(end):
            return path[: -len(end)]
    return path


def image_ids(ds) -> list[str]:
    """Return one id per dataset sample: its ``image_path`` with no image file ending (V-E-24).

    The file name alone is not unique. For example, ``patient1/study1/view1_frontal.jpg``
    and ``patient2/study1/view1_frontal.jpg`` both have ``view1_frontal``.
    """
    return [_cut_image_ending(p) for p in dataset_rows(ds)["image_path"].astype(str).values]


def png_name(image_id: str) -> str:
    """Return a flat PNG file name for ``image_id``. ``a/b/c`` gives ``a__b__c.png``."""
    return image_id.replace("\\", "/").strip("/").replace("/", "__") + ".png"


def patient_ids_of(ds) -> np.ndarray | None:
    """Return one patient id per dataset sample, or ``None`` when the table has no ``patient_id``.

    Raises:
        ValueError: some rows have a NaN ``patient_id`` (D-32, Q-5). A NaN id is a harmonizer bug
            (D-01). The message gives the number of these rows and their first image paths.
    """
    df = dataset_rows(ds)
    if "patient_id" not in df.columns:
        return None
    missing = df["patient_id"].isna()
    if missing.any():
        first = df.loc[missing, "image_path"].astype(str).head(3).tolist()
        raise ValueError(f"{int(missing.sum())} rows have no patient_id (NaN), for example {first}. "
                         "Each row needs a patient_id (D-01). Fix the harmonizer, or drop these rows.")
    return df["patient_id"].astype(str).values


def group_ids(ds) -> np.ndarray:
    """Return one group id per dataset sample, for the patient splits and the bootstrap (D-32).

    It is the ``patient_id``. With no ``patient_id`` column, each image is its own
    patient: its id is its ``image_path``. Then the bootstrap draws images (V-E-33).
    """
    pids = patient_ids_of(ds)
    return pids if pids is not None else dataset_rows(ds)["image_path"].astype(str).values


def carve_val(
    sub: np.ndarray, rng, patient_ids: np.ndarray, val_fraction: float
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split the train rows ``sub`` into fit rows and validation rows, by patient.

    The validation part has about ``val_fraction`` of the patients in ``sub``
    (D-29). All images of one patient go to the same part.

    Args:
        sub: row indices of the train rows.
        rng: numpy Generator. It gives the seed of the split.
        patient_ids: one patient id per row of the whole dataset.
        val_fraction: part of the patients for validation, for example 0.1.

    Returns:
        ``(fit_idx, val_idx)``. ``val_idx`` is ``None`` when ``val_fraction``
        is 0, ``sub`` has fewer than 2 rows, or all rows have one patient.
    """
    if val_fraction <= 0.0 or len(sub) < 2:
        return sub, None
    sub_pids = np.asarray(patient_ids)[sub]
    if len(np.unique(sub_pids)) < 2:
        return sub, None
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )
    fit_pos, val_pos = next(gss.split(np.arange(len(sub)), groups=sub_pids))
    return sub[fit_pos], sub[val_pos]


def bootstrap_indices(n_rows: int, n_bootstrap: int, seed: int, groups=None):
    """Yield the row indices of each bootstrap sample of a test set (D-32).

    Each sample draws patients with replacement and keeps all rows of each
    drawn patient. The metric is still computed over rows (images).

    Example: a patient with 20 follow-up images is drawn as one unit, so the 20
    images go in or out together. For a test set with one row per patient, the
    draws are the same as a draw of rows.

    Args:
        n_rows: number of test rows.
        n_bootstrap: number of samples.
        seed: seed of the generator. A negative seed is folded to an unsigned
            32-bit value.
        groups: one patient id per row. ``None`` draws rows.

    Yields:
        np.ndarray of row indices. Its length changes from sample to sample
        when the patients have different numbers of rows.
    """
    if n_bootstrap <= 0:
        return
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    if groups is None:
        for _ in range(n_bootstrap):
            yield rng.integers(0, n_rows, size=n_rows)
        return
    groups = np.asarray(groups)
    if len(groups) != n_rows:
        raise ValueError(f"groups has {len(groups)} ids for {n_rows} rows.")
    # Number the patients in the order of their first row. Then, with one row
    # per patient, patient i is row i, and the draws equal a draw of rows.
    _, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    rank = np.empty(len(first), dtype=np.int64)
    rank[np.argsort(first)] = np.arange(len(first))
    patient = rank[inverse.ravel()]
    rows_by_patient = np.argsort(patient, kind="stable")
    counts = np.bincount(patient)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    n_patients = len(counts)
    for _ in range(n_bootstrap):
        drawn = rng.integers(0, n_patients, size=n_patients)
        c = counts[drawn]
        # Row position t of drawn patient p is rows_by_patient[starts[p] + t].
        offsets = np.repeat(starts[drawn] - np.cumsum(c) + c, c) + np.arange(c.sum())
        yield rows_by_patient[offsets]


# The run kind is a part of the torch seed, so a k-fold run, a fixed-split run
# and the final model never share a seed.
RUN_KFOLD, RUN_FIXED, RUN_FINAL = 0, 1, 2


def seed_torch(*parts: int) -> None:
    """Seed the global torch RNG from integer parts (R-code-3).

    The torch RNG sets the head weights, the DataLoader shuffle and dropout.
    Call it before each run (fold or seed, and n_train point), so the same
    ``base_seed`` gives the same result. The evaluator's numpy RNG is not
    used, so the train subsamples do not change.

    Example: ``seed_torch(base_seed, RUN_KFOLD, fold, n_train)``. A negative
    part (for example ``n_train=-1``) is folded to an unsigned 32-bit value.
    """
    state = np.random.SeedSequence([int(p) & 0xFFFFFFFF for p in parts]).generate_state(1)
    torch.manual_seed(int(state[0]))
