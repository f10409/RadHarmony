# RadHarmony-ViT: LeJEPA 2D Pretraining & Linear Probe Eval

Reference LeJEPA self-supervised pretraining of a NaFlex ViT-Base on harmonized
chest X-ray datasets (CheXpert, MIMIC-CXR, ChestX-ray14) via RadHarmony, plus a
linear-probe evaluation script on VinDr-CXR.

## Setup

```bash
cd model
uv sync
```

## Pretraining — `train.py`

Multi-dataset pretraining at 256 resolution:

```bash
python train.py \
  --epochs 100 \
  --batch_size 64 \
  --lr 1e-4 \
  --devices 0,1 \
  --project radharmony-LeJEPA \
  --test_name naflex_full_run \
  --chexpert_dir /path/to/CheXpert-v1.0/train/ \
  --mimic_dir /path/to/MIMIC-CXR/files/ \
  --mimic_label_csv /path/to/mimic-cxr-2.0.0-chexpert.csv \
  --chestxray14_dir /path/to/ChestX-ray14/ \
  --cache_dir /path/to/cache/256
```

Resolution adaptation (continue pretraining at 512 from a 256 checkpoint).
`--init_ckpt` loads weights only and resets optimizer/scheduler/epoch counter:

```bash
python train.py \
  --init_ckpt ./weights/naflex_full_run_epoch=52_val/pred_loss=0.1484.ckpt \
  --img_size 512 \
  --epochs 5 \
  --lr 1e-5 \
  --lam_warmup 600 \
  --batch_size 64 \
  --devices 0,1 \
  --gradient_checkpointing \
  --project radharmony-LeJEPA \
  --test_name naflex_512_adapt \
  --chexpert_dir /path/to/CheXpert-v1.0/train/ \
  --mimic_dir /path/to/MIMIC-CXR/files/ \
  --mimic_label_csv /path/to/mimic-cxr-2.0.0-chexpert.csv \
  --chestxray14_dir /path/to/ChestX-ray14/ \
  --cache_dir /path/to/cache/512 \
  --accumulate_grad_batches 2 
```

MIMIC-only ablation — pass only `--mimic_dir` (+ `--mimic_label_csv`) and drop
the other dataset flags.

Scaling study — subsample the combined training pool with
`--sample_fraction 0.1` or `--sample_count 50000`. Fix step budget with
`--max_steps` to compare runs at matched compute.

Resume a full run (optimizer/scheduler/epoch preserved) with `--ckpt`
instead of `--init_ckpt`.

### Key flags

| Flag | Meaning |
| --- | --- |
| `--chexpert_dir` / `--mimic_dir` / `--chestxray14_dir` | Dataset roots — any subset enables only those datasets |
| `--mimic_label_csv` | Path to `mimic-cxr-2.0.0-chexpert.csv` (CheXpert labels for MIMIC) |
| `--cache_dir` | MONAI `PersistentDataset` cache root (per-dataset subdirs auto-created) |
| `--img_size` | Input resolution (256 default, 512 for high-res adaptation) |
| `--probe_label` | Online-probe target label (default `pneumothorax`) |
| `--init_ckpt` / `--ckpt` | Weights-only init vs. full-state resume |
| `--gradient_checkpointing` | Saves memory for 512-res / large batches |
| `--accumulate_grad_batches` | Effective-batch multiplier |
| `--sample_fraction` / `--sample_count` | Subsample training pool for scaling studies |
| `--max_steps` | Fixed step budget (overrides `--epochs`) |

## Linear-probe evaluation — `eval_linear_probe_vindr.py`

5-fold CV linear probe on the combined VinDr-CXR train+test pool, sweeping
training-pool sizes. Reports per-label and macro AUROC/AUPRC, writes CSVs and
sweep plots.

```bash
python eval_linear_probe_vindr.py \
  --ckpt ./weights/naflex_512_adapt/epoch_4_loss0.1070.ckpt \
  --vindr_train_dir /path/to/vindr-cxr/1.0.0/train/ \
  --vindr_test_dir  /path/to/vindr-cxr/1.0.0/test/ \
  --device 0 --img_size 512 \
  --n_folds 5 \
  --n_train_samples 1500 3750 7500 11250 14400 \
  --cache_dir /path/to/cache/eval/ \
  --output_dir ./eval_results/512_adapt_run_vindr_kfold
```

Add `--stratified_subsample` to sample train indices via iterative
stratification on the multi-label matrix instead of uniform random.
`--img_size` defaults to the checkpoint's training resolution; override to
re-extract embeddings at a different size.

Outputs:

- `vindr_fold_results.csv` — one row per (n_train, label, fold)
- `vindr_linear_probe_summary.csv` — per-label means ± stds + macro average
- `vindr_auroc_vs_ntrain.png`, `vindr_auprc_vs_ntrain.png`

> Note: this script mixes the 3-rater consensus test split into the
> cross-validation pool, so numbers are **not** directly comparable to the
> official VinDr-CXR paper protocol.
