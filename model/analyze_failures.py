"""Analyze per-sample failures across all three linear-probe backbones.

For each test sample × label, finds cases where every backbone predicted
the wrong class.  Exports a Parquet file ready for Gradio / pandas analysis.

Usage
-----
  cd model
  RADHARMONY_OUTPUT_DIR=/path/to/outputs .venv/bin/python analyze_failures.py

It reads the caches and results of eval_linear_probe_taix.py under RADHARMONY_OUTPUT_DIR
(default: ./outputs) and writes to <RADHARMONY_OUTPUT_DIR>/taix_failure_analysis.
"""
from __future__ import annotations

import os, sys, pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ── paths ──────────────────────────────────────────────────────────────────
# The data paths file reader is in <repo>/scripts.
_SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from dataset_paths import paths  # noqa: E402

_TAIX = paths("taix_ray_512")
TAIX_IMG_DIR  = _TAIX["base_image_dir"]
TAIX_CSV      = _TAIX["csv_path"]
# Caches and results go under RADHARMONY_OUTPUT_DIR (default: ./outputs).
OUTPUT_ROOT   = os.environ.get("RADHARMONY_OUTPUT_DIR") or "./outputs"
CACHE_ROOT    = os.path.join(OUTPUT_ROOT, "taix_lp_cache")
RESULTS_ROOT  = os.path.join(OUTPUT_ROOT, "taix_lp_results")
OUTPUT_DIR    = os.path.join(OUTPUT_ROOT, "taix_failure_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BACKBONES = ["chexagent", "medsiglip", "biomed_clip"]

LR_KWARGS = dict(
    penalty="l2",
    max_iter=10000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=0,
)


# ── helpers ────────────────────────────────────────────────────────────────

def load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_test_df() -> pd.DataFrame:
    """Return the test-split DataFrame in the same order as the embedding pkl."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from radharmony.dataset.taix_ray import TAIXRay512Dataset

    # Harmonise once to get the split column + stable ordering.
    full_ds = TAIXRay512Dataset(
        base_image_dir=TAIX_IMG_DIR,
        csv_path=TAIX_CSV,
        output_cls=True,
        label_mode="binary",
        cache_dir=os.path.join(CACHE_ROOT, "chexagent", "full"),   # reuse any cached harmonised df
    )
    df = full_ds.get_harmonized_df()
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    return test_df


def get_thresholds(backbone: str, label_names: list[str]) -> np.ndarray:
    """Load per-label thresholds (mean over seeds) from results_summary."""
    summary = pd.read_csv(os.path.join(RESULTS_ROOT, backbone, "results_summary.csv"))
    summary = summary[summary["label"] != "macro_average"].set_index("label")
    return np.array([summary.loc[lbl, "threshold_mean"] for lbl in label_names])


def fit_probes(X_train: np.ndarray, y_train: np.ndarray) -> list:
    """Fit one LogisticRegression per label column (seed 0)."""
    probes = []
    n_labels = y_train.shape[1]
    for k in range(n_labels):
        y_k = y_train[:, k].astype(int)
        clf = LogisticRegression(**LR_KWARGS)
        if len(np.unique(y_k)) >= 2:
            clf.fit(X_train, y_k)
        probes.append(clf)
    return probes


def predict_proba(probes: list, X_test: np.ndarray) -> np.ndarray:
    """Return (N_test, L) probability matrix."""
    probs = np.zeros((X_test.shape[0], len(probes)), dtype=np.float32)
    for k, clf in enumerate(probes):
        if hasattr(clf, "classes_"):
            probs[:, k] = clf.predict_proba(X_test)[:, 1]
        else:
            probs[:, k] = np.nan
    return probs


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print("Loading test-set DataFrame …")
    test_df = get_test_df()
    n_test = len(test_df)
    print(f"  {n_test:,} test samples")

    # Build full image paths (harmonized df has filename only in image_path)
    test_df["image_path"] = test_df["image_path"].apply(
        lambda fn: os.path.join(TAIX_IMG_DIR, fn)
    )

    # Collect per-backbone predictions
    all_probs  = {}   # backbone -> (N, L) float32
    all_preds  = {}   # backbone -> (N, L) bool
    label_names = None

    for backbone in BACKBONES:
        print(f"\n[{backbone}] Loading embeddings …")
        train_pkl = load_pkl(os.path.join(CACHE_ROOT, backbone, "embeddings", "taix.train.pkl"))
        test_pkl  = load_pkl(os.path.join(CACHE_ROOT, backbone, "embeddings", "taix.test.pkl"))

        X_train, y_train = train_pkl["feats"], train_pkl["labels"]
        X_test,  y_test  = test_pkl["feats"],  test_pkl["labels"]

        if label_names is None:
            label_names = test_pkl["labels_names"]
            y_true = y_test.astype(int)   # same for all backbones
            assert y_true.shape[0] == n_test, \
                f"Mismatch: test_df has {n_test} rows but embedding has {y_true.shape[0]}"

        assert X_test.shape[0] == n_test, \
            f"[{backbone}] embedding rows {X_test.shape[0]} ≠ test_df rows {n_test}"

        print(f"  train {X_train.shape}, test {X_test.shape}")
        print(f"  fitting {len(label_names)} LR probes (seed 0) …")
        probes = fit_probes(X_train, y_train)

        probs = predict_proba(probes, X_test)
        thresholds = get_thresholds(backbone, label_names)   # (L,)
        preds = (probs >= thresholds[None, :]).astype(int)   # (N, L)

        all_probs[backbone] = probs
        all_preds[backbone] = preds
        print(f"  done  (threshold range {thresholds.min():.3f}–{thresholds.max():.3f})")

    print("\nBuilding per-sample failure DataFrame …")

    # Base columns: metadata + true labels
    rows = test_df[["study_id", "patient_id", "image_path", "age", "sex", "study_date"]].copy()
    n_labels = len(label_names)

    for k, lbl in enumerate(label_names):
        rows[f"true_{lbl}"] = y_true[:, k]

    # Per-backbone predicted probs + binary preds
    for backbone in BACKBONES:
        short = backbone.replace("biomed_clip", "biomedclip")
        for k, lbl in enumerate(label_names):
            rows[f"prob_{short}_{lbl}"]  = all_probs[backbone][:, k]
            rows[f"pred_{short}_{lbl}"]  = all_preds[backbone][:, k]

    # Error flags per (sample, label, backbone)
    for k, lbl in enumerate(label_names):
        true_k = y_true[:, k]
        for backbone in BACKBONES:
            short = backbone.replace("biomed_clip", "biomedclip")
            rows[f"err_{short}_{lbl}"] = (all_preds[backbone][:, k] != true_k).astype(np.int8)

    # All-model-failed flag per label
    for k, lbl in enumerate(label_names):
        true_k = y_true[:, k]
        all_wrong = np.ones(n_test, dtype=bool)
        for backbone in BACKBONES:
            all_wrong &= (all_preds[backbone][:, k] != true_k)
        rows[f"all_fail_{lbl}"] = all_wrong.astype(np.int8)

    # Summary: number of labels where all models failed
    fail_cols = [f"all_fail_{lbl}" for lbl in label_names]
    rows["n_labels_all_fail"] = rows[fail_cols].sum(axis=1)

    # Any-label all-fail flag
    rows["any_label_all_fail"] = (rows["n_labels_all_fail"] > 0).astype(np.int8)

    # ── statistics ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("All-model failure rates per label")
    print(f"{'='*60}")
    for lbl in label_names:
        col = f"all_fail_{lbl}"
        n_fail = rows[col].sum()
        rate   = n_fail / n_test * 100
        print(f"  {lbl:<35} {n_fail:>6,} / {n_test:,}  ({rate:.1f}%)")

    hard = rows[rows["any_label_all_fail"] == 1]
    print(f"\n  Total samples with ≥1 all-fail label: {len(hard):,} / {n_test:,} ({len(hard)/n_test*100:.1f}%)")

    # ── export ─────────────────────────────────────────────────────────────
    # Full table (all test samples)
    out_full = os.path.join(OUTPUT_DIR, "taix_lp_predictions.parquet")
    rows.to_parquet(out_full, index=False)
    print(f"\nFull prediction table → {out_full}")

    # Failure-only table (any label all-fail)
    out_fail = os.path.join(OUTPUT_DIR, "taix_lp_all_fail_cases.parquet")
    hard.to_parquet(out_fail, index=False)
    print(f"All-fail cases only   → {out_fail}")

    # CSV version for quick inspection
    out_csv = os.path.join(OUTPUT_DIR, "taix_lp_all_fail_cases.csv")
    hard.to_csv(out_csv, index=False)
    print(f"CSV (same content)    → {out_csv}")

    # Per-label failure summary
    summary_rows = []
    for lbl in label_names:
        col = f"all_fail_{lbl}"
        lbl_df = rows[rows[col] == 1]
        summary_rows.append({
            "label": lbl,
            "n_all_fail": len(lbl_df),
            "pct_all_fail": round(len(lbl_df) / n_test * 100, 2),
            "prevalence_pct": round(y_true[:, label_names.index(lbl)].mean() * 100, 2),
        })
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(OUTPUT_DIR, "failure_summary_by_label.csv"), index=False
    )
    print(f"Per-label summary     → {OUTPUT_DIR}/failure_summary_by_label.csv")
    print(f"\nDone.")


if __name__ == "__main__":
    main()
