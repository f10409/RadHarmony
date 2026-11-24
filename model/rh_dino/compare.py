"""Paired bootstrap of two runs of evaluate.py: B - A, with a 95% CI and a two-sided p-value.

vindr (``--task vindr``): both runs use the same 5 seeds, so each seed trains on the same 1,500 images. Each seed
also uses the same patient bootstrap draws (``LinearProbeEvaluator``: bootstrap seed = bootstrap_seed + seed index
x 10,000 + n_train). So row (seed, bootstrap) of A and of B score the same test images, and B - A is a paired
difference. The tool stops if the two runs do not have the same train and test images in the same order.
Output: one row per label: A and B (mean of the 5 full-test probes), diff (B - A), seed_min and seed_max (the
lowest and highest full-test diff of one seed), ci_lo and ci_hi (2.5 and 97.5 percentiles of the paired bootstrap
differences, all seeds together), p and n_boot.

siim (``--task siim``): the per-image Dice and GIoU of the images with pneumothorax (per_image.csv). The tool
draws ChestX-ray14 patients with replacement (5,000 times, seed 0) and averages the paired differences.

p = 2 x min((k_le + 1) / (n + 1), (k_ge + 1) / (n + 1)), at most 1, where k_le and k_ge count the bootstrap
differences <= 0 and >= 0 among the n resamples.

Usage (from model/rh_dino/):
    python compare.py --task vindr OUT/vindr/raddino OUT/vindr/rh_dino
    python compare.py --task siim OUT/siim/raddino OUT/siim/rh_dino
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def p_value(x: np.ndarray) -> float:
    """Two-sided bootstrap p-value of the differences ``x`` (see the module docstring)."""
    n = len(x)
    return min(1.0, 2 * (min((x <= 0).sum(), (x >= 0).sum()) + 1) / (n + 1))


def _image_paths(run: Path, split: str) -> np.ndarray:
    with open(run / f"embeddings.{split}.pkl", "rb") as f:
        return np.asarray(pickle.load(f)["image_paths"])


def vindr_diffs(a: Path, b: Path, metric: str) -> pd.DataFrame:
    """Return one row per label of the paired VinDr-CXR comparison (see the module docstring)."""
    for split in ("train", "test"):
        if not np.array_equal(_image_paths(a, split), _image_paths(b, split)):
            raise SystemExit(f"{split} images differ between {a} and {b}: the rows are not paired")
    key = ["seed", "bootstrap", "label"]
    ra, rb = (pd.read_csv(run / "results.csv").set_index(key)[metric] for run in (a, b))
    if not ra.index.sort_values().equals(rb.index.sort_values()):
        raise SystemExit("the two results.csv files have different (seed, bootstrap, label) rows")
    diffs = (rb - ra).reset_index(name="diff")
    full, boot = diffs[diffs.bootstrap == -1], diffs[diffs.bootstrap >= 0]
    out = []
    for lab, g in boot.groupby("label"):
        x = g["diff"].dropna().to_numpy()
        seeds = full.loc[full.label == lab, "diff"]
        out.append({
            "label": lab,
            "A": ra.xs(-1, level="bootstrap").xs(lab, level="label").mean(),
            "B": rb.xs(-1, level="bootstrap").xs(lab, level="label").mean(),
            "diff": seeds.mean(), "seed_min": seeds.min(), "seed_max": seeds.max(),
            "ci_lo": np.percentile(x, 2.5), "ci_hi": np.percentile(x, 97.5),
            "p": p_value(x), "n_boot": len(x),
        })
    return pd.DataFrame(out)


def siim_diffs(a: Path, b: Path, n_boot: int) -> pd.DataFrame:
    """Return one row per metric (dice_pos, giou_pos) of the paired SIIM-ACR comparison."""
    pa, pb = (pd.read_csv(run / "per_image.csv") for run in (a, b))
    m = pa.merge(pb, on=["image_id", "patient_id", "positive", "fold"], suffixes=("_a", "_b"))
    if not len(m) == len(pa) == len(pb):
        raise SystemExit(f"per_image.csv rows differ: {len(pa)}, {len(pb)}, {len(m)} paired")
    m = m[m["positive"].astype(bool)]
    pat = m["patient_id"].to_numpy()
    patients = np.unique(pat)
    rows_of = {p: np.where(pat == p)[0] for p in patients}
    out = []
    for met in ("dice_pos", "giou_pos"):
        rng = np.random.default_rng(0)
        d = (m[met + "_b"] - m[met + "_a"]).to_numpy()
        x = np.array([d[np.concatenate([rows_of[p] for p in rng.choice(patients, len(patients))])].mean()
                      for _ in range(n_boot)])
        out.append({"metric": met, "A": m[met + "_a"].mean(), "B": m[met + "_b"].mean(), "diff": d.mean(),
                    "ci_lo": np.percentile(x, 2.5), "ci_hi": np.percentile(x, 97.5), "p": p_value(x),
                    "n_images": len(m), "n_boot": n_boot})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["vindr", "siim"])
    ap.add_argument("run_a", type=Path, help="result folder of run A: <out>/<task>/<name> of evaluate.py")
    ap.add_argument("run_b", type=Path, help="result folder of run B: <out>/<task>/<name> of evaluate.py")
    ap.add_argument("--metric", default="auprc", choices=["auprc", "auroc"], help="vindr only")
    ap.add_argument("--n-boot", type=int, default=5000, help="siim only: the number of patient resamples")
    a = ap.parse_args()
    if a.task == "vindr":
        print(f"{a.metric}: A = {a.run_a.name}, B = {a.run_b.name}, diff = B - A")
        table = vindr_diffs(a.run_a, a.run_b, a.metric)
    else:
        print(f"SIIM-ACR, images with pneumothorax: A = {a.run_a.name}, B = {a.run_b.name}, diff = B - A")
        table = siim_diffs(a.run_a, a.run_b, a.n_boot)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
