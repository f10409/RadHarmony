"""Find the ChestX-ray14 (NIH) file of each SIIM-ACR image, so that pretraining can leave them out.

SIIM-ACR images are ChestX-ray14 images (1024 x 1024, JPEG-compressed DICOM) with new names. Each image is
reduced to a 32 x 32 fingerprint. For each SIIM-ACR image, the NIH image with the highest correlation is its
match (rank 1). Of the next 4 candidates, the tool also writes:
- a candidate with correlation > 0.99: the same image or a near copy (NIH has a few identical files);
- when the match is not sure: a candidate within 0.005 of the rank-1 correlation. Example: a SIIM-ACR image
  (male, 16) has rank 1 at 0.9624 (male, 33, a different person) and its true file at 0.9609.
``sure``: corr > 0.99 and corr - corr_2nd > 0.02. Every SIIM-ACR image gives at least one row.

The released siim_acr_in_chestxray14.csv is the output of this tool, without the sex, age and view columns.
evaluate.py uses its rank-1 rows to group the SIIM-ACR images by ChestX-ray14 patient.

Usage (from model/rh_dino/; one GPU, about 3 min):
    python match_siim_acr_nih.py --nih-csv NIH_CSV --nih-dir NIH_DIR --siim-dir SIIM_DIR --out OUT_CSV
NIH_CSV is Data_Entry_2017.csv of ChestX-ray14, NIH_DIR holds its images_*/images/ folders, and SIIM_DIR holds
dicom-images-train/ and dicom-images-test/ of SIIM-ACR.
Output: nih_file, siim_file, rank, corr, corr_2nd, sure, meta_agrees and the sex, age and view of both.
"""
import argparse
import glob
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

N = 32  # fingerprint size
K = 5   # candidates per SIIM-ACR image


def fingerprint(img: Image.Image) -> np.ndarray:
    """32 x 32 box-filtered gray image, z-scored, as a float16 vector."""
    x = np.asarray(img.convert("L").resize((N, N), Image.BOX), dtype=np.float32).ravel()
    return ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float16)


def nih_one(p):
    """Fingerprint of one NIH PNG (zeros if it does not load)."""
    try:
        return fingerprint(Image.open(p))
    except Exception:
        return np.zeros(N * N, np.float16)


def siim_one(p):
    """Fingerprint and (sex, age, view) of one SIIM-ACR DICOM (zeros and empty strings if it does not load)."""
    import pydicom

    try:
        d = pydicom.dcmread(p)
        meta = (str(d.get("PatientSex", "")), str(d.get("PatientAge", "")), str(d.get("ViewPosition", "")))
        return fingerprint(Image.fromarray(d.pixel_array)), meta
    except Exception:
        return np.zeros(N * N, np.float16), ("", "", "")


def run(fn, files, name):
    """Run fn on all files with 32 processes and print the time."""
    t0 = time.time()
    with Pool(32) as pool:
        out = list(pool.imap(fn, files, chunksize=64))
    print(f"{name}: {len(files):,} images in {time.time() - t0:.0f} s", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nih-csv", required=True, help="ChestX-ray14 Data_Entry_2017.csv")
    ap.add_argument("--nih-dir", required=True, help="ChestX-ray14 folder with images_*/images/*.png")
    ap.add_argument("--siim-dir", required=True, help="SIIM-ACR folder with dicom-images-train/ and dicom-images-test/")
    ap.add_argument("--out", required=True, type=Path, help="output CSV")
    args = ap.parse_args()
    nih = pd.read_csv(args.nih_csv)
    pattern = os.path.join(args.nih_dir, "images_*", "images", "*.png")
    found = {os.path.basename(f): f for f in glob.glob(pattern)}
    nih = nih[nih["Image Index"].isin(found)].reset_index(drop=True)
    f_nih = np.stack(run(nih_one, [found[x] for x in nih["Image Index"]], "ChestX-ray14"))

    siim_files = sorted(f for k in ("dicom-images-train", "dicom-images-test")
                        for f in glob.glob(os.path.join(args.siim_dir, k, "*", "*", "*.dcm")))
    r = run(siim_one, siim_files, "SIIM-ACR")
    f_siim = np.stack([a for a, _ in r])
    meta = pd.DataFrame([m for _, m in r], columns=["sex", "age", "view"])

    # Pearson correlation of every pair (both are z-scored), on the GPU in blocks
    a = torch.from_numpy(f_siim).float().cuda() / (N * N)
    b = torch.from_numpy(f_nih).float().cuda()
    top = []
    for i in range(0, len(a), 1024):
        v, j = (a[i:i + 1024] @ b.T).topk(K, dim=1)
        top.append(torch.cat([v, j.float()], 1).cpu())  # float32 holds the indices (< 2^24) exactly
    top = torch.cat(top).numpy()
    corr, idx = top[:, :K], top[:, K:].astype(int)
    base = meta.assign(siim_file=[os.path.relpath(f, args.siim_dir) for f in siim_files], corr_2nd=corr[:, 1])
    base["sure"] = (corr[:, 0] > 0.99) & (corr[:, 0] - corr[:, 1] > 0.02)
    parts = []
    for k in range(K):
        keep = np.ones(len(base), bool) if k == 0 else (
            (corr[:, k] > 0.99) | (~base["sure"].to_numpy() & (corr[:, k] >= corr[:, 0] - 0.005)))
        parts.append(base[keep].assign(rank=k + 1, corr=corr[keep, k], nih_idx=idx[keep, k]))
    out = pd.concat(parts, ignore_index=True)
    n = nih.iloc[out["nih_idx"]].reset_index(drop=True)
    out = out.assign(nih_file=n["Image Index"], nih_sex=n["Patient Gender"], nih_age=n["Patient Age"],
                     nih_view=n["View Position"]).drop(columns="nih_idx")
    out["meta_agrees"] = ((out["sex"] == out["nih_sex"])
                          & (pd.to_numeric(out["age"], errors="coerce") == out["nih_age"])
                          & (out["view"] == out["nih_view"]))
    cols = ["nih_file", "siim_file", "rank", "corr", "corr_2nd", "sure", "meta_agrees", "sex", "age", "view",
            "nih_sex", "nih_age", "nih_view"]
    out[cols].to_csv(args.out, index=False, float_format="%.6f")
    print(f"{len(siim_files):,} SIIM-ACR images, {out['nih_file'].nunique():,} NIH files, "
          f"sure {int(base['sure'].sum()):,}, sex/age/view agree {int(out['meta_agrees'].sum()):,} of {len(out):,} rows"
          f" -> {args.out}")


if __name__ == "__main__":
    main()
