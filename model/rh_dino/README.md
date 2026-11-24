# RH-DINO

RH-DINO is a chest radiograph encoder from the case study of the RadHarmony paper. This folder has the code
that evaluates it. The training code is not released. The supplementary material of the paper lists all
training settings.

## The model

- **Architecture:** ViT-B/16 with 4 register tokens (86 million parameters), in the Hugging Face `DINOv3ViTModel`
  format.
- **Start:** the public DINOv3 weights `facebook/dinov3-vitb16-pretrain-lvd1689m`, trained on natural images.
- **Training:** continued self-supervised training with the DINOv3 objective (DINO and iBOT losses, KoLeo
  weight 0.02).
  - Stage 1: 300,000 steps at 256 pixels, batch size 512.
  - Stage 2: 6,112 steps at 512 pixels, batch size 128.
- **Data:** 782,369 frontal chest radiographs from six public datasets: MIMIC-CXR, ChestX-ray14, CheXpert
  (the DICOM images of CheXpert Plus), PadChest, BRAX and ReXGradient-160K. RadHarmony loads all of them.
  The ChestX-ray14 images that are also in SIIM-ACR are not in the training data.
- **Input:** 512 x 512 pixels. Scale each image to its 0.5 and 99.5 intensity percentiles (`scaling.py`).
  Then use the DINOv3 preprocessing. Use the CLS token as the image feature.

## Results

The models are frozen. VinDr-CXR: a linear probe on the CLS token (1,500 training images, 5 seeds, 7
findings), tested on the official 3,000-image test split. SIIM-ACR: a 1 x 1 conv head on the patch features,
5-fold cross-validation grouped by patient. The 95% confidence intervals come from a bootstrap over
patients.

| Model | VinDr-CXR test macro AUPRC [95% CI] | VinDr-CXR test macro AUROC | SIIM-ACR Dice [95% CI] |
|---|---|---|---|
| Original DINOv3 ViT-B/16 (512) | 0.318 [0.289, 0.360] | 0.827 | 0.225 [0.205, 0.245] |
| RAD-DINO (518) | 0.467 [0.441, 0.499] | 0.903 | 0.313 [0.291, 0.334] |
| RH-DINO (512) | 0.445 [0.408, 0.485] | 0.892 | 0.349 [0.325, 0.372] |

Paired differences of RH-DINO and RAD-DINO:

- VinDr-CXR AUPRC: -0.022 (95% CI -0.052 to 0.006, p = 0.14). The difference is not significant.
- SIIM-ACR Dice: +0.035 (95% CI 0.026 to 0.046, p < 0.001).

RAD-DINO was trained on all ChestX-ray14 images, so it saw the SIIM-ACR images. RH-DINO did not.

## Get the weights

The weights are not public yet. We put them on Hugging Face when the paper is published. They are under the
DINOv3 License, because RH-DINO is a derivative of the DINOv3 weights. The Hugging Face model repository has a
copy of the license (`LICENSE.md`).

## Use the model

Install RadHarmony with the `dinov3` and `raddino` extras and scikit-image
(`uv pip install -e ".[dinov3,raddino]" scikit-image`). The `raddino` extra and scikit-image are only for
the evaluation. Run this code from this folder. `RH_DINO_DIR` is the folder of the weights.

```python
import torch
from evaluate import make_recipe

transform, encoder, _ = make_recipe("dinov3", RH_DINO_DIR, seg=False, device="cuda",
                                    scaling="percentile", output_keys={"img"})
x = transform({"img": "image.dcm"})["img"]  # (3, 512, 512); PNG and JPEG also work
with torch.no_grad():
    feature = encoder(x[None].to("cuda"))  # (1, 768): the CLS token
```

## Reproduce the evaluation

The paper evaluates three models with the same protocol: RH-DINO, RAD-DINO and the original DINOv3. Download
VinDr-CXR (PhysioNet) and SIIM-ACR Pneumothorax (Kaggle) yourself. `VINDR_DIR` is the VinDr-CXR release
folder (`vindr-cxr/1.0.0`, with `train/`, `test/` and `annotations/`). `SIIM_CSV` is `train-rle.csv`, and
`SIIM_DICOM_DIR` is `dicom-images-train/`. `eval.ipynb` runs all the steps below for the three models.

### The three models

Each model uses the preprocessing of its own code. Give the same `--backbone`, `--hub` and `--scaling` to
every step of a model.

| Model | Arguments | Input |
|---|---|---|
| RH-DINO | `--backbone dinov3 --hub RH_DINO_DIR --scaling percentile` | 512 |
| RAD-DINO | `--backbone raddino` | 518 |
| Original DINOv3 | `--backbone dinov3 --hub DINOV3_512_DIR` | 512 |

- **RAD-DINO:** the public checkpoint `microsoft/rad-dino`. The RadHarmony recipe loads it with the `rad-dino`
  package and its image processor (resize and center crop to 518 pixels).
- **Original DINOv3:** `facebook/dinov3-vitb16-pretrain-lvd1689m` with the RadHarmony DINOv3 recipe (min-max
  scaling). Its Hugging Face folder sets the input size to 224. Make a copy with the size 512:

  ```bash
  python make_dinov3_512.py DINOV3_512_DIR
  ```

### The steps

Do steps 1 to 3 for each model. The examples are for RH-DINO. For another model, change the model arguments
and `--name` (for example `--backbone raddino --name raddino`).

1. Choose the probe C on the validation split (3,000 fixed training images). The test set does not choose C.

   ```bash
   python evaluate.py --task vindr_val --backbone dinov3 --hub RH_DINO_DIR --scaling percentile \
       --C 0.01 0.1 1 10 100 1000 --vindr-dir VINDR_DIR --name rh_dino --out OUT --cache CACHE
   ```

2. Test with that C.

   ```bash
   python evaluate.py --task vindr --backbone dinov3 --hub RH_DINO_DIR --scaling percentile \
       --C-from OUT/vindr_val/rh_dino --vindr-dir VINDR_DIR --name rh_dino --out OUT --cache CACHE
   ```

3. Run the segmentation probe.

   ```bash
   python evaluate.py --task siim --backbone dinov3 --hub RH_DINO_DIR --scaling percentile \
       --siim-csv SIIM_CSV --siim-dicom-dir SIIM_DICOM_DIR --name rh_dino --out OUT --cache CACHE
   ```

4. Compare two models with a paired bootstrap.

   ```bash
   python compare.py --task vindr OUT/vindr/raddino OUT/vindr/rh_dino
   python compare.py --task siim OUT/siim/raddino OUT/siim/rh_dino
   ```

In the paper, the grid chose C = 10 for RH-DINO and RAD-DINO and C = 100 for the original DINOv3.

## Files

| File | What it does |
|---|---|
| `evaluate.py` | VinDr-CXR linear probe (`vindr`, `vindr_val`) and SIIM-ACR segmentation probe (`siim`) |
| `scaling.py` | The intensity scaling of the RH-DINO pretraining (0.5 and 99.5 percentiles) |
| `compare.py` | Paired bootstrap of two runs: difference, 95% CI and two-sided p-value |
| `make_dinov3_512.py` | Makes the folder of the original DINOv3 with the input size 512 |
| `demo.ipynb` | Loads RH-DINO and shows its image feature, a PCA map and a similarity map of the patch features |
| `eval.ipynb` | Runs the evaluation of the paper for the three models and shows the results next to the paper |
| `siim_acr_in_chestxray14.csv` | The ChestX-ray14 file of each SIIM-ACR image. `evaluate.py` uses it to group the SIIM-ACR images by patient. |
| `match_siim_acr_nih.py` | Makes `siim_acr_in_chestxray14.csv` from the images (32 x 32 fingerprints and correlation) |

## Intended use

RH-DINO is for research. It is not a medical device. Do not use it for clinical decisions.

## License

The code in this folder is under the Apache License 2.0, as the rest of RadHarmony. The RH-DINO weights are
under the DINOv3 License. RH-DINO is built with DINOv3 by Meta (Siméoni et al., DINOv3).
