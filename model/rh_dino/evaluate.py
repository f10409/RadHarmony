"""Evaluate a frozen backbone as in the RadHarmony paper: VinDr-CXR linear probe and SIIM-ACR segmentation probe.

VinDr-CXR (RAD-DINO protocol, RadHarmony ``LinearProbeEvaluator``): a logistic regression on the CLS token of
1,500 train images (a new sample for each of 5 seeds), tested on the official 3,000 test images, for 7 findings,
with AUPRC, AUROC and a patient bootstrap. ``vindr_val`` uses the train split only: a fixed random set of 3,000
train images replaces the test images, and the 1,500 come from the other 12,000. Use it to choose settings (the
probe C), so that the test set is used only for the final models.

SIIM-ACR (``LinearProbeSegEvaluator``): a 1 x 1 conv head on the frozen patch features, 5-fold cross-validation
grouped by patient on the train split (the test split has no masks). SIIM-ACR images are ChestX-ray14 images and
SIIM-ACR has no real patient id, so the patient is the ChestX-ray14 patient from ``siim_acr_in_chestxray14.csv``.
From the saved masks: Dice on images with pneumothorax, GIoU on these images (convex hull, Rezatofighi et al.
2019) and Dice on all images (an empty prediction on an empty mask gives NaN and is left out), with a patient
bootstrap over all folds.

Backbones: ``raddino`` (RadHarmony recipe, input 518) or ``dinov3`` with ``--hub``: a Hugging Face id or a local
folder, for example the RH-DINO folder (its ``preprocessor_config.json`` sets the input size, 512).
``--scaling percentile`` gives a dinov3 model the intensity scaling of the RH-DINO pretraining (0.5 and 99.5
percentiles) in place of the min-max scaling of the RadHarmony recipe. Use it for RH-DINO.

Probe C: ``--C`` with several values on ``vindr_val`` runs a grid and picks the best macro AUPRC. ``--C-from``
then uses that C on ``vindr``, so the test set never chooses C.

Usage (from model/rh_dino/, in an environment with RadHarmony installed):
    python evaluate.py --task vindr_val --backbone dinov3 --hub RH_DINO_DIR --scaling percentile \\
        --C 0.01 0.1 1 10 100 --vindr-dir VINDR_DIR --name rh_dino --out OUT --cache CACHE
    python evaluate.py --task vindr --backbone dinov3 --hub RH_DINO_DIR --scaling percentile \\
        --C-from OUT/vindr_val/rh_dino --vindr-dir VINDR_DIR --name rh_dino --out OUT --cache CACHE
    python evaluate.py --task siim --backbone dinov3 --hub RH_DINO_DIR --scaling percentile \\
        --siim-csv SIIM_CSV --siim-dicom-dir SIIM_DICOM_DIR --name rh_dino --out OUT --cache CACHE
Output: OUT/<task>/<name>/ with args.json, embeddings.*.pkl, results.csv and results_summary.csv. With several
--C: also C<value>/ (one probe each), c_grid.csv and best_C.json; the top results are those of the best C.
For siim: also Pred/, GT/, per_image.csv and seg_summary.csv. Compare two runs with compare.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from scaling import percentile_255

# RAD-DINO Table 1(a): LO, CM, PL-T, AE, PF, TB, PE
VINDR_LABELS = ["lung_opacity", "cardiomegaly", "pleural_thickening", "aortic_enlargement",
                "pulmonary_fibrosis", "tuberculosis", "pleural_effusion"]
SEG_METRICS = ["dice_pos", "giou_pos", "dice_all"]
SEG_FOLDS = 5
VINDR_VAL_SIZE = 3000
SIIM_IN_NIH = Path(__file__).resolve().parent / "siim_acr_in_chestxray14.csv"


def make_recipe(name: str, hub: str | None, seg: bool, device: str, scaling: str = "minmax",
                output_keys: set[str] | None = None):
    """Return (transform, encoder, prep) of a RadHarmony recipe.

    ``output_keys``: the sample keys of the transform. Default: {"img", "mask"} if ``seg``, else the recipe
    default {"img", "cls"}. Use {"img"} to embed images that have no labels.
    ``scaling`` (dinov3 only): "minmax" = the RadHarmony DINOv3 recipe; "percentile" = the same recipe with
    the pretraining intensity scaling (``scaling.percentile_255``) in place of its min-max scaling.
    ``prep`` names the image cache: "raddino", "dinov3_<size>" or "dinov3_<size>_pct".
    """
    from radharmony.evaluator.backbones import dinov3 as dinov3_recipe
    from radharmony.evaluator.backbones import make_raddino

    keys = output_keys or ({"img", "mask"} if seg else None)
    if name == "raddino":
        return (*make_raddino(device=device, output_keys=keys), "raddino")
    pp = Path(hub, "preprocessor_config.json")
    size = json.loads(pp.read_text())["size"]["height"] if pp.exists() else 224
    if scaling == "minmax":
        return (*dinov3_recipe.make_dinov3(device=device, hub=hub, output_keys=keys), f"dinov3_{size}")
    # make_dinov3 has no scaling option. It puts the module name display_255 into its loader when it builds, so
    # we swap the name for this call only; the built loader keeps percentile_255. The check stops a silent
    # min-max run if the recipe changes.
    if not hasattr(dinov3_recipe, "display_255"):
        raise RuntimeError("make_dinov3 no longer uses display_255; update make_recipe")
    old, dinov3_recipe.display_255 = dinov3_recipe.display_255, percentile_255
    try:
        recipe = dinov3_recipe.make_dinov3(device=device, hub=hub, output_keys=keys)
    finally:
        dinov3_recipe.display_255 = old
    return (*recipe, f"dinov3_{size}_pct")  # its own image cache


def run_vindr(a, transform, encoder, prep: str, out: Path) -> pd.DataFrame:
    """VinDr-CXR linear probe (test or val split). Returns the summary table (of the best C for a grid).

    One C: results in ``out``. Several C: one probe per C in ``out/C<value>/``, all on the same embeddings, then
    ``c_grid.csv`` and ``best_C.json`` (best macro AUPRC; on a tie the first C given), and the best results
    copied to ``out``.
    """
    from radharmony.dataset import VinDrCXRTestDataset, VinDrCXRTrainDataset
    from radharmony.evaluator.classification import LinearProbeEvaluator
    from radharmony.evaluator.sampling import dataset_rows

    ds = {split: cls(base_image_dir=os.path.join(a.vindr_dir, split), transform=transform,
                     cache_dir=f"{a.cache}/{prep}_vindr_{split}", output_cls=True)
          for split, cls in (("train", VinDrCXRTrainDataset), ("test", VinDrCXRTestDataset))}
    if a.task == "vindr_val":
        # A fixed val set of 3,000 train images (the same for every backbone); the probe trains on the others.
        rows = dataset_rows(ds["train"]).reset_index(drop=True)
        is_val = np.zeros(len(rows), bool)
        is_val[np.random.default_rng(0).choice(len(rows), VINDR_VAL_SIZE, replace=False)] = True
        ds["train"], ds["test"] = (
            VinDrCXRTrainDataset(base_image_dir=os.path.join(a.vindr_dir, "train"), harmonized_df=rows[m],
                                 transform=transform, cache_dir=f"{a.cache}/{prep}_vindr_train", output_cls=True)
            for m in (~is_val, is_val))
    if len(a.C) == 1:  # a grid of an earlier run in this folder must not be read by a later --C-from
        for f in ("best_C.json", "c_grid.csv"):
            (out / f).unlink(missing_ok=True)
    summaries = {}
    for c in a.C:
        o = out if len(a.C) == 1 else out / f"C{c:g}"  # a grid: one folder per C, the same embeddings
        o.mkdir(exist_ok=True)
        ev = LinearProbeEvaluator(
            encoder, train_dataset=ds["train"], test_dataset=ds["test"], labels=VINDR_LABELS,
            n_train_samples=[1500], n_seeds=5, n_bootstrap=a.n_bootstrap, probe_kwargs={"C": c},
            val_fraction=0.0,  # fit on all 1,500 images, as RAD-DINO (the threshold metrics are then not used)
            device=a.device, batch_size=a.batch_size,
            num_workers=a.workers, autocast_dtype=torch.bfloat16, embedding_cache=str(out / "embeddings"),
            output_dir=str(o),
        )
        ev.save_results(ev.evaluate())
        summaries[c] = pd.read_csv(o / "results_summary.csv")
    if len(a.C) == 1:
        return summaries[a.C[0]]
    # The best C (macro AUPRC); its results are also copied to `out`. Use it on test with --C-from.
    grid = pd.DataFrame([{"C": c, **s.set_index("label").loc["macro_average", ["auprc_mean", "auroc_mean"]]}
                         for c, s in summaries.items()])
    grid.to_csv(out / "c_grid.csv", index=False)
    best = float(grid.loc[grid["auprc_mean"].idxmax(), "C"])
    (out / "best_C.json").write_text(json.dumps({"C": best}))
    for f in ("results.csv", "results_summary.csv"):
        shutil.copy(out / f"C{best:g}" / f, out / f)
    print(grid.to_string(index=False), f"\nbest C = {best:g}")
    return summaries[best]


def giou(pred: np.ndarray, gt: np.ndarray) -> float:
    """GIoU of two binary masks; C is the convex hull of both (NaN if both are empty)."""
    from skimage.morphology import convex_hull_image

    union = pred | gt
    if not union.any():
        return np.nan
    c = convex_hull_image(union).sum()
    return (pred & gt).sum() / union.sum() - (c - union.sum()) / c


def seg_scores(out: Path, image_ids: list[str], groups: np.ndarray, n_bootstrap: int) -> pd.DataFrame:
    """Per-image Dice and GIoU from the saved masks of all folds, then a patient bootstrap.

    Writes per_image.csv and seg_summary.csv in ``out``. Returns the summary.
    """
    from radharmony.evaluator.sampling import bootstrap_indices, png_name

    where = {}  # png name -> fold folder (each image is tested in one fold)
    for i in range(SEG_FOLDS):
        d = f"fold{i}_n-1"
        for f in os.listdir(out / "Pred" / d):
            if f in where:
                raise RuntimeError(f"{f} is in {where[f]} and {d}")
            where[f] = d
    if len(where) != len(image_ids):
        raise RuntimeError(f"{len(where)} predicted masks for {len(image_ids)} images; delete {out} and run again")
    rows = []
    for iid, g in zip(image_ids, groups, strict=True):
        fn = png_name(iid)
        p = np.asarray(Image.open(out / "Pred" / where[fn] / fn)) > 0
        t = np.asarray(Image.open(out / "GT" / where[fn] / fn)) > 0
        s = p.sum() + t.sum()
        dice = 2 * (p & t).sum() / s if s else np.nan  # an empty prediction on an empty mask: NaN, left out
        pos = bool(t.any())
        rows.append({"image_id": iid, "patient_id": g, "fold": where[fn], "positive": pos,
                     "dice_all": dice, "dice_pos": dice if pos else np.nan, "giou_pos": giou(p, t) if pos else np.nan})
    per = pd.DataFrame(rows)
    per.to_csv(out / "per_image.csv", index=False)

    # The same seed for every backbone, so all backbones get the same patient samples (paired).
    boot = {m: [] for m in SEG_METRICS}
    for idx in bootstrap_indices(len(per), n_bootstrap, seed=0, groups=per["patient_id"].to_numpy()):
        sample = per.iloc[idx]
        for m in SEG_METRICS:
            boot[m].append(np.nanmean(sample[m]))
    summary = pd.DataFrame([{
        "metric": m, "mean": np.nanmean(per[m]), "ci_lo": np.percentile(boot[m], 2.5),
        "ci_hi": np.percentile(boot[m], 97.5), "n_images": int(per[m].notna().sum()),
    } for m in SEG_METRICS])
    summary.to_csv(out / "seg_summary.csv", index=False)
    return summary


def run_siim(a, transform, encoder, prep: str, out: Path) -> pd.DataFrame:
    """SIIM-ACR segmentation linear probe (5 folds). Returns the Dice / GIoU summary."""
    import skimage.morphology  # noqa: F401  (giou needs it; fail now, not after the 5 folds)
    from radharmony.dataset import SIIMACRPTXTrainDataset
    from radharmony.evaluator.sampling import group_ids, image_ids
    from radharmony.evaluator.segmentation import LinearProbeSegEvaluator
    from radharmony.harmonizer import SIIMACRPTXTrainHarmonizer

    df = SIIMACRPTXTrainHarmonizer(csv_path=a.siim_csv, dicom_dir=a.siim_dicom_dir).harmonize(
        mask_output_dir=f"{a.cache}/siim_masks", mask_num_cores=a.workers)
    # The SIIM-ACR patient_id is a study folder (one per image). Use the ChestX-ray14 patient (rank-1 match).
    m = pd.read_csv(SIIM_IN_NIH).query("rank == 1")
    nih = dict(zip(m["siim_file"].map(os.path.basename), m["nih_file"].str[:8], strict=True))
    pid = df["image_path"].map(lambda x: nih.get(os.path.basename(x)))
    if pid.isna().any():
        raise RuntimeError(f"{int(pid.isna().sum())} SIIM-ACR images have no ChestX-ray14 match")
    df["patient_id"] = "nih_" + pid
    print(f"SIIM-ACR: {len(df):,} images, {df['patient_id'].nunique():,} ChestX-ray14 patients")
    ds = SIIMACRPTXTrainDataset(base_image_dir=a.siim_dicom_dir, harmonized_df=df, transform=transform,
                                cache_dir=f"{a.cache}/{prep}_siim", output_mask=True)
    ev = LinearProbeSegEvaluator(
        encoder, dataset=ds, num_classes=2, n_folds=SEG_FOLDS, epochs=20, lr=1e-3, weight_decay=0.05,
        class_weights=[1.0, 5.0], batch_size=16, save_predictions=True, early_stop_metric="dice",
        early_stop_patience=5, val_fraction=0.1, device=a.device, num_workers=a.workers,
        autocast_dtype=torch.bfloat16, base_seed=0, output_dir=str(out),
    )
    df = ev.evaluate()
    ev.save_results(df)
    return seg_scores(out, image_ids(ds), group_ids(ds), a.n_bootstrap)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["vindr", "vindr_val", "siim"])
    ap.add_argument("--backbone", required=True, choices=["raddino", "dinov3"])
    ap.add_argument("--hub", default=None, help="dinov3 only: Hugging Face id or local model folder (RH-DINO)")
    ap.add_argument("--name", required=True, help="result folder name, for example rh_dino")
    ap.add_argument("--out", required=True, help="result root")
    ap.add_argument("--cache", required=True, help="image cache root (one folder per input size and dataset)")
    ap.add_argument("--vindr-dir",
                    help="vindr tasks: the VinDr-CXR folder (vindr-cxr/1.0.0) with train/, test/ and annotations/")
    ap.add_argument("--siim-csv", help="siim: SIIM-ACR train-rle.csv")
    ap.add_argument("--siim-dicom-dir", help="siim: SIIM-ACR dicom-images-train folder")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--scaling", default="minmax", choices=["minmax", "percentile"],
                    help="dinov3 only. minmax: the RadHarmony recipe. percentile: 0.5/99.5 percentiles to 0..255, "
                         "as in the RH-DINO pretraining (own image cache, _pct)")
    ap.add_argument("--C", type=float, nargs="+", default=None,
                    help="VinDr probe C (default 1). Several values (vindr_val only): a grid, one folder C<value>/ "
                         "each, plus c_grid.csv and best_C.json")
    ap.add_argument("--C-from", default=None,
                    help="vindr only, in place of --C: a vindr_val result folder; use the C in its best_C.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64, help="VinDr embedding batch size (siim uses 16)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.backbone == "dinov3" and not a.hub:
        ap.error("--backbone dinov3 needs --hub")
    if a.backbone == "raddino" and a.hub:
        ap.error("--hub is for --backbone dinov3 only")
    if a.hub and os.path.isdir(a.hub):  # a local folder: one spelling, so the --C-from check can compare it
        a.hub = os.path.abspath(a.hub)
    if a.scaling != "minmax" and a.backbone != "dinov3":
        ap.error("--scaling is for --backbone dinov3 only")
    if a.task == "siim" and not (a.siim_csv and a.siim_dicom_dir):
        ap.error("--task siim needs --siim-csv and --siim-dicom-dir")
    if a.task != "siim" and not a.vindr_dir:
        ap.error(f"--task {a.task} needs --vindr-dir")
    if a.task == "siim" and (a.C or a.C_from):
        ap.error("--C and --C-from are for the vindr tasks only")
    if a.C and a.C_from:
        ap.error("give --C or --C-from, not both")
    if a.C_from:
        f = Path(a.C_from, "best_C.json")
        if not f.exists():
            ap.error(f"{f} does not exist: run a --C grid on vindr_val first")
        # The C must come from a run with the same features.
        v = json.loads(Path(a.C_from, "args.json").read_text())
        for k in ("backbone", "hub", "scaling"):
            if v.get(k) != getattr(a, k):
                ap.error(f"--C-from used --{k} {v.get(k)}, not {getattr(a, k)}")
        a.C = [json.loads(f.read_text())["C"]]
    a.C = a.C or [1.0]
    if len(a.C) > 1 and a.task != "vindr_val":
        ap.error("a --C grid is for vindr_val only (the test set must not choose C)")
    if len({f"{c:g}" for c in a.C}) != len(a.C):
        ap.error("--C values must be different (also after rounding to 6 digits)")

    # MONAI writes each cache file to a temp folder, then moves it. On the same disk the move is one rename, so a
    # reader never sees a half-written file (a copy from /tmp to another disk can be read half-done).
    os.makedirs(a.cache, exist_ok=True)
    tempfile.tempdir = a.cache
    out = Path(a.out, a.task, a.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(a), indent=2))
    transform, encoder, prep = make_recipe(a.backbone, a.hub, a.task == "siim", a.device, a.scaling)
    summary = (run_siim if a.task == "siim" else run_vindr)(a, transform, encoder, prep, out)
    with pd.option_context("display.width", 200, "display.max_columns", 12):
        print(summary)


if __name__ == "__main__":
    main()
