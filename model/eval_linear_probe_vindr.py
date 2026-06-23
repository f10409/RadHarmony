"""Linear probe evaluation on VinDr-CXR with k-fold CV over the combined pool.

Protocol:
  - Pool: concatenation of the 15k official train + 3k official test images.
  - K-fold split (default 5) over the full pool. Each fold's held-out split
    is the evaluation set (~3.6k); the remaining ~14.4k is the *train pool*
    for that fold.
  - For each n_train in --n_train_samples, subsample n_train indices from the
    fold's train pool (uniform or iterative-stratified), fit a per-label
    logistic-regression linear probe on a frozen LeJEPA2D backbone, and
    evaluate AUROC/AUPRC on the fold's held-out split.
  - Reports per-label metrics (mean ± std across folds) plus a macro average.

Note: unlike the paper protocol this mixes the 3-rater consensus test labels
into the train pool, so numbers are NOT comparable to eval_linear_probe_vindr_paper.py.
This script is an internal scaling ablation, not a literature benchmark.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from radharmony.dataset import (
    RadiologyTransform2D,
    VinDrCXRTestDataset,
    VinDrCXRTrainDataset,
)

from modules import LeJEPA2D


DEFAULT_LABELS = [
    "lung_opacity",
    "cardiomegaly",
    "pleural_thickening",
    "aortic_enlargement",
    "pulmonary_fibrosis",
    "tuberculosis",
    "pleural_effusion",
]

LABEL_ABBR = {
    "lung_opacity": "LO",
    "cardiomegaly": "CM",
    "pleural_thickening": "PL-T",
    "aortic_enlargement": "AE",
    "pulmonary_fibrosis": "PF",
    "tuberculosis": "TB",
    "pleural_effusion": "PE",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="VinDr-CXR linear probe: 5 seeds × 1500 train samples, fixed test."
    )
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--vindr_train_dir", type=str, required=True)
    p.add_argument("--vindr_train_csv", type=str, default=None)
    p.add_argument("--vindr_test_dir", type=str, required=True)
    p.add_argument("--vindr_test_csv", type=str, default=None)
    p.add_argument("--cache_dir", type=str, default="./cache/eval")
    p.add_argument(
        "--probe_labels",
        type=str,
        nargs="+",
        default=DEFAULT_LABELS,
        help="Target labels to probe (macro average reported across them).",
    )
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument(
        "--n_train_samples",
        type=int,
        nargs="+",
        default=[1500, 3750, 7500, 11250, 14400],
        help="Train set sizes to sweep. One linear probe is fit per "
        "(n_train, seed, label) triple.",
    )
    p.add_argument(
        "--stratified_subsample",
        action="store_true",
        help="Sample train indices via iterative stratification on the "
        "multi-label matrix instead of uniform random.",
    )
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--device", type=int, default=0)
    p.add_argument(
        "--output_dir", type=str, default="./eval_results/linear_probe_vindr"
    )
    return p.parse_args()


def stratified_subsample_indices(labels_all, n_samples, seed):
    """Pick `n_samples` row indices from `labels_all` via iterative stratification.

    Splits the pool into k ≈ pool / n_samples folds using IterativeStratification
    on the multi-label matrix; returns fold `seed % k` (trimmed to n_samples).
    """
    from skmultilearn.model_selection import IterativeStratification

    pool_size = len(labels_all)
    k = max(2, round(pool_size / n_samples))
    proportions = [1.0 / k] * k
    X = np.zeros((pool_size, 1), dtype=np.float32)
    strat = IterativeStratification(
        n_splits=k, order=1, sample_distribution_per_fold=proportions
    )
    folds = [test_idx for _, test_idx in strat.split(X, labels_all)]
    fold = folds[seed % k]
    if len(fold) > n_samples:
        rng = np.random.default_rng(seed)
        fold = rng.choice(fold, size=n_samples, replace=False)
    return np.asarray(fold)


def _resolve_targets(label_cols, probe_labels):
    resolved = []
    for want in probe_labels:
        match = next((c for c in label_cols if c.lower() == want.lower()), None)
        if match is None:
            raise ValueError(f"'{want}' not found in dataset LABEL_COLS")
        resolved.append(match)
    return resolved


def _make_dataset(DatasetCls, base_dir, csv_path, transform, cache_subdir, probe_labels):
    kwargs = dict(
        base_image_dir=base_dir,
        transform=transform,
        output_cls=True,
        cache_dir=cache_subdir,
    )
    if csv_path:
        kwargs["csv_path"] = csv_path
    ds = DatasetCls(**kwargs)
    targets = _resolve_targets(ds.LABEL_COLS, probe_labels)
    ds.LABEL_COLS = targets
    ds.verify_images()
    return ds, targets


@torch.no_grad()
def extract_embeddings(model, loader, device, desc="Extracting"):
    model.eval()
    all_feats, all_labels = [], []
    for batch in tqdm(loader, desc=desc):
        imgs = batch["img"].to(device=device, dtype=torch.bfloat16)
        labels = batch["cls"]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            feats = model.backbone(imgs)
        all_feats.append(feats.float().cpu().numpy())
        all_labels.append(labels.float().numpy())
    feats = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    if labels.ndim == 1:
        labels = labels[:, None]
    return feats, labels


def plot_sweep(summary_df, n_train_list, targets, output_dir):
    import matplotlib.pyplot as plt

    plot_labels = list(targets) + ["macro_average"]
    for metric, ylabel, fname in [
        ("mean_auroc", "AUROC", "vindr_auroc_vs_ntrain.png"),
        ("mean_auprc", "AUPRC", "vindr_auprc_vs_ntrain.png"),
    ]:
        std_col = metric.replace("mean_", "std_")
        fig, ax = plt.subplots(figsize=(7, 5))
        for label in plot_labels:
            sub = summary_df[summary_df["label"] == label].sort_values("n_train")
            if sub.empty:
                continue
            xs = sub["n_train"].to_numpy()
            ys = sub[metric].to_numpy()
            es = sub[std_col].to_numpy()
            abbr = LABEL_ABBR.get(label, "Agg." if label == "macro_average" else label)
            is_macro = label == "macro_average"
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                label=abbr,
                marker="o",
                linewidth=2.5 if is_macro else 1.0,
                color="black" if is_macro else None,
                alpha=1.0 if is_macro else 0.35,
                zorder=3 if is_macro else 2,
            )
        ax.set_xlabel("# training samples")
        ax.set_ylabel(ylabel)
        ax.set_title(f"VinDr-CXR linear probe: {ylabel} vs training set size")
        ax.set_xticks(n_train_list)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        out_path = os.path.join(output_dir, fname)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Plot saved to {out_path}")


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.device}")

    print(f"Loading checkpoint from {args.ckpt} …")
    model = LeJEPA2D.load_from_checkpoint(
        args.ckpt, map_location="cpu", weights_only=False
    )
    model = model.to(device)
    model.eval()

    img_size = args.img_size if args.img_size is not None else model.hparams.img_size
    print(f"Using img_size={img_size}")
    transform = RadiologyTransform2D(
        img_size=img_size,
        output_keys={"img", "cls"},
    ).get_transform()

    loader_kwargs = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    train_ds, train_targets = _make_dataset(
        VinDrCXRTrainDataset,
        args.vindr_train_dir,
        args.vindr_train_csv,
        transform,
        os.path.join(args.cache_dir, "vindr_train"),
        args.probe_labels,
    )
    test_ds, test_targets = _make_dataset(
        VinDrCXRTestDataset,
        args.vindr_test_dir,
        args.vindr_test_csv,
        transform,
        os.path.join(args.cache_dir, "vindr_test"),
        args.probe_labels,
    )
    if train_targets != test_targets:
        raise ValueError(
            f"Train/test label order mismatch: {train_targets} vs {test_targets}"
        )
    # get_datasets(n_splits=None) returns a torch-compatible PersistentDataset
    # and also sorts LABEL_COLS in place, so re-read label order afterward.
    train_torch_ds = train_ds.get_datasets(n_splits=None, num_cores=4)
    test_torch_ds = test_ds.get_datasets(n_splits=None, num_cores=4)
    targets = list(train_ds.LABEL_COLS)
    if list(test_ds.LABEL_COLS) != targets:
        raise ValueError("Train/test LABEL_COLS order mismatch after get_datasets")
    print(f"Probe labels (final order): {targets}")

    n_train_orig = len(train_torch_ds)
    n_test_orig = len(test_torch_ds)
    print(f"Official train: {n_train_orig:,} | Official test: {n_test_orig:,}")

    os.makedirs(args.output_dir, exist_ok=True)

    train_loader = DataLoader(train_torch_ds, **loader_kwargs)
    test_loader = DataLoader(test_torch_ds, **loader_kwargs)

    train_feats_part, train_labels_part = extract_embeddings(
        model, train_loader, device, desc="Train embeddings"
    )
    test_feats_part, test_labels_part = extract_embeddings(
        model, test_loader, device, desc="Test embeddings"
    )
    # Combined pool: official train + official test concatenated.
    feats_all = np.concatenate([train_feats_part, test_feats_part], axis=0)
    labels_all = np.concatenate([train_labels_part, test_labels_part], axis=0)
    n_pool = len(feats_all)
    print(
        f"Combined pool: {n_pool:,} images | "
        f"positives per label: "
        f"{dict(zip(targets, labels_all.sum(0).astype(int).tolist()))}"
    )

    n_train_list = sorted(set(int(n) for n in args.n_train_samples))
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.base_seed)
    approx_train_pool = n_pool - n_pool // args.n_folds
    for n in n_train_list:
        if n > approx_train_pool:
            raise ValueError(
                f"n_train_samples={n} exceeds per-fold train pool "
                f"(~{approx_train_pool})"
            )
    print(
        f"{args.n_folds}-fold CV on combined pool. "
        f"Sweeping n_train_samples: {n_train_list}"
    )

    per_fold_rows = []  # one row per (n_train, label, fold)
    label_summary_rows = []  # one row per (n_train, label) + macro

    # fold_splits: list of (train_pool_idx, test_idx) arrays.
    fold_splits = list(kf.split(np.arange(n_pool)))

    # Pre-compute subsampled train indices for each (n_train, fold).
    # For stratified mode we call IterativeStratification on the fold's
    # train-pool label matrix once per (n_train, fold).
    sub_idx: dict[tuple[int, int], np.ndarray] = {}
    for n_train in n_train_list:
        for fold_i, (train_pool_idx, _) in enumerate(fold_splits):
            fold_seed = args.base_seed + fold_i
            if args.stratified_subsample:
                sub_local = stratified_subsample_indices(
                    labels_all[train_pool_idx], n_train, fold_seed
                )
                picked = train_pool_idx[sub_local]
                prev = labels_all[picked].mean(0)
                prev_str = ", ".join(
                    f"{LABEL_ABBR.get(t, t)}={p:.3f}"
                    for t, p in zip(targets, prev.tolist())
                )
                print(
                    f"n_train={n_train} fold={fold_i} stratified prev: {prev_str}"
                )
            else:
                rng = np.random.default_rng(fold_seed + 1000 * n_train)
                sub_local = rng.choice(
                    len(train_pool_idx), size=n_train, replace=False
                )
                picked = train_pool_idx[sub_local]
            sub_idx[(n_train, fold_i)] = picked

    for n_train in n_train_list:
        for k, label in enumerate(targets):
            label_auroc, label_auprc = [], []
            for fold_i, (_, test_idx) in enumerate(fold_splits):
                tr_idx = sub_idx[(n_train, fold_i)]
                tr_feats = feats_all[tr_idx]
                tr_labels = labels_all[tr_idx, k]
                te_feats = feats_all[test_idx]
                te_labels = labels_all[test_idx, k]

                if len(np.unique(tr_labels)) < 2:
                    print(
                        f"[skip] n_train={n_train} label={label} fold={fold_i}: "
                        f"only one class in train sample"
                    )
                    continue

                scaler = StandardScaler()
                tr_feats_s = scaler.fit_transform(tr_feats)
                te_feats_s = scaler.transform(te_feats)

                clf = LogisticRegression(
                    max_iter=10000, solver="lbfgs", class_weight="balanced"
                )
                clf.fit(tr_feats_s, tr_labels)
                probs = clf.predict_proba(te_feats_s)[:, 1]
                auroc = (
                    roc_auc_score(te_labels, probs)
                    if len(np.unique(te_labels)) > 1
                    else float("nan")
                )
                auprc = average_precision_score(te_labels, probs)

                label_auroc.append(auroc)
                label_auprc.append(auprc)
                per_fold_rows.append(
                    {
                        "n_train": n_train,
                        "label": label,
                        "abbr": LABEL_ABBR.get(label, label),
                        "fold": fold_i,
                        "auroc": auroc,
                        "auprc": auprc,
                        "train_pos": int(tr_labels.sum()),
                        "train_size": len(tr_labels),
                        "test_pos": int(te_labels.sum()),
                        "test_size": len(te_labels),
                    }
                )

            mean_auroc = (
                float(np.nanmean(label_auroc)) if label_auroc else float("nan")
            )
            std_auroc = (
                float(np.nanstd(label_auroc)) if label_auroc else float("nan")
            )
            mean_auprc = (
                float(np.nanmean(label_auprc)) if label_auprc else float("nan")
            )
            std_auprc = (
                float(np.nanstd(label_auprc)) if label_auprc else float("nan")
            )
            abbr = LABEL_ABBR.get(label, label)
            print(
                f"n_train={n_train:>6} {abbr:>5} ({label}): "
                f"AUROC {mean_auroc:.4f} ± {std_auroc:.4f} | "
                f"AUPRC {mean_auprc:.4f} ± {std_auprc:.4f}"
            )
            label_summary_rows.append(
                {
                    "n_train": n_train,
                    "label": label,
                    "abbr": abbr,
                    "mean_auroc": mean_auroc,
                    "std_auroc": std_auroc,
                    "mean_auprc": mean_auprc,
                    "std_auprc": std_auprc,
                }
            )

        this_rows = [r for r in label_summary_rows if r["n_train"] == n_train]
        label_mean_auroc = np.array([r["mean_auroc"] for r in this_rows])
        label_mean_auprc = np.array([r["mean_auprc"] for r in this_rows])
        macro_auroc = float(np.nanmean(label_mean_auroc))
        macro_auprc = float(np.nanmean(label_mean_auprc))
        label_summary_rows.append(
            {
                "n_train": n_train,
                "label": "macro_average",
                "abbr": "Agg.",
                "mean_auroc": macro_auroc,
                "std_auroc": float(np.nanstd(label_mean_auroc)),
                "mean_auprc": macro_auprc,
                "std_auprc": float(np.nanstd(label_mean_auprc)),
            }
        )
        print(
            f"n_train={n_train:>6} Agg. (macro over {len(targets)} labels): "
            f"AUROC {macro_auroc:.4f} | AUPRC {macro_auprc:.4f}"
        )

    fold_path = os.path.join(args.output_dir, "vindr_fold_results.csv")
    pd.DataFrame(per_fold_rows).to_csv(fold_path, index=False)

    summary_df = pd.DataFrame(label_summary_rows)
    summary_path = os.path.join(args.output_dir, "vindr_linear_probe_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Results saved to {fold_path} and {summary_path}")

    plot_sweep(summary_df, n_train_list, targets, args.output_dir)


if __name__ == "__main__":
    main()
