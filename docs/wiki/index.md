# RadHarmony Wiki

RadHarmony unifies **both** public radiological datasets (CXR, CT, MRI) **and** their downstream evaluation pipelines under one consistent interface — so the same backbone can be benchmarked across every dataset, the same dataset can be probed with every evaluator, and every run produces the same on-disk schema. The result is foundation-model evaluations that are **scalable** (one new dataset or backbone reuses the entire pipeline) and **reproducible** (no per-dataset, per-evaluator glue code to drift).

## What problem does it solve?

Each public radiology dataset ships its own CSV format, label vocabulary, image format, target definitions, and folder layout — and each foundation model ships its own pretraining recipe, image preprocessor, and forward-pass signature. RadHarmony eliminates both axes of glue code:

- **Harmonizer + Dataset layer** — every dataset becomes a MONAI `PersistentDataset` with a consistent data dict (`sample["img"]` plus whichever of `cls`, `bbox`, `mask`, `report` the dataset supports), regardless of source format.
- **Evaluator + Backbone layer** — every evaluator (probe / zero-shot / fine-tune / segmentation) takes any RadHarmony dataset and any image encoder and emits a standardized `pd.DataFrame` of metrics plus a paired summary CSV. Backbone recipes (`make_raddino`, `make_biomed_clip`, `make_medsiglip`, …) drop in a `(transform, encoder)` pair ready to pair with any dataset.

Because the two layers share a common dict contract, swapping the dataset or the backbone is a one-line change — no per-(dataset × backbone × evaluator) rewriting.

## Quick code teaser

Load a harmonized dataset:

```python
import torch
from radharmony.dataset import CheXpertTrainDataset

ds = CheXpertTrainDataset(base_image_dir="/data/CheXpert-v1.0/train/", output_cls=True, dtype=torch.float32)
train_ds, val_ds = ds.get_datasets(n_splits=5)
sample = train_ds[0]   # {"img": Tensor(1,224,224), "cls": Tensor(14,)}
```

Plug a backbone into any evaluator across the supported datasets:

```python
from radharmony.evaluator import LinearProbeEvaluator
from radharmony.evaluator.backbones import make_raddino
from radharmony.dataset import VinDrCXRTrainDataset, VinDrCXRTestDataset

transform, encoder = make_raddino(device="cuda")
train_ds = VinDrCXRTrainDataset(base_image_dir="/data/VinDr-CXR/", transform=transform, output_cls=True)
test_ds  = VinDrCXRTestDataset (base_image_dir="/data/VinDr-CXR/", transform=transform, output_cls=True)

ev = LinearProbeEvaluator(encoder, train_dataset=train_ds, test_dataset=test_ds, n_seeds=3, n_bootstrap=100)
df = ev.evaluate()           # standardized AUROC / AUPRC / F1 / … DataFrame
ev.save_results(df)          # results.csv + results_summary.csv
```

## Evaluators

| Family | Evaluators | Mode |
|--------|------------|------|
| Classification probes | [LinearProbeEvaluator](evaluator/linear_probe.md), [KNNProbeEvaluator](evaluator/knn_probe.md), [SVMProbeEvaluator](evaluator/svm_probe.md), [PrototypeProbeEvaluator](evaluator/prototype_probe.md) | k-fold / fixed-split |
| Vision-language | [ZeroShotEvaluator](evaluator/zero_shot.md) | test-only |
| End-to-end | [FinetuneEvaluator](evaluator/finetune.md) | k-fold / fixed-split |
| Segmentation probes | [LinearProbeSegEvaluator](evaluator/linear_probe_seg.md), [ConvProbeSegEvaluator](evaluator/conv_probe_seg.md), [UPerNetSegEvaluator](evaluator/upernet_seg.md) | k-fold / fixed-split |

Every evaluator shares the same constructor surface (`dataset=` for k-fold, `train_dataset=` + `test_dataset=` for fixed-split, `n_seeds` × `n_bootstrap` for variance estimation), the same metric panel, and the same on-disk output (`results.csv` + `results_summary.csv` with per-row and per-label mean / std / 95 % CI). See the [Evaluator API](evaluator/index.md) overview for the full table and the [Backbones](backbones/index.md) tab for the foundation-model recipes.

## Wiki pages

| Page | Description |
|------|-------------|
| [Quickstart](quickstart.md) | Install, launch the app, load your first dataset |
| [Datasets](datasets.md) | Full inventory table with links to per-dataset pages |
| [Acquiring Datasets](acquiring-datasets.md) | Download commands for every dataset, grouped by access method |
| [Dataset API](api.md) | Constructors, splits, data dict keys, caching, harmonizer usage |
| [Transforms](transforms.md) | 2D/3D pipelines, augmentations, HU windowing, dtype |
| [Evaluator API](evaluator/index.md) | Linear / k-NN / SVM / prototype probes, zero-shot, fine-tune, segmentation probes |
| [Backbones](backbones/index.md) | Drop-in factory functions for the supported foundation models |
| [Custom Backbones](backbones/custom_backbones.md) | Wrap your own model into the evaluator pipeline |
| [App Guide](app.md) | Launch the Gradio UI, UI walkthrough, remote access |
| [Architecture](architecture.md) | Layer diagram, data flow, key invariants |

## Per-dataset pages

| Dataset | Modality | Page |
|---------|----------|------|
| CheXpert | CXR | [chexpert.md](datasets/chexpert.md) |
| CheXpert-Plus | CXR | [chexpert_plus.md](datasets/chexpert_plus.md) |
| MIMIC-CXR (DICOM) | CXR | [mimic_cxr.md](datasets/mimic_cxr.md) |
| MIMIC-CXR-JPG | CXR | [mimic_cxr_jpg.md](datasets/mimic_cxr_jpg.md) |
| ChestX-ray14 | CXR | [chestxray14.md](datasets/chestxray14.md) |
| PadChest | CXR | [padchest.md](datasets/padchest.md) |
| ReXGradient-160K | CXR | [rexgradient.md](datasets/rexgradient.md) |
| VinDr-CXR | CXR | [vindr_cxr.md](datasets/vindr_cxr.md) |
| SIIM-ACR Pneumothorax | CXR | [siim_acr_ptx.md](datasets/siim_acr_ptx.md) |
| SIIM COVID-19 | CXR | [siim_covid19.md](datasets/siim_covid19.md) |
| RSNA Pneumonia | CXR | [rsna_pneumonia.md](datasets/rsna_pneumonia.md) |
| RSNA Pneumonia (Kaggle) | CXR | [rsna_pneumonia_kaggle.md](datasets/rsna_pneumonia_kaggle.md) |
| RSNA PE Detection <sup class="beta">beta</sup> | CT | [rsna_pe_detection.md](datasets/rsna_pe_detection.md) |
| RSNA Pediatric Bone Age <sup class="beta">beta</sup> | Radiograph | [rsna_bone_age.md](datasets/rsna_bone_age.md) |
| RSNA 2024 Lumbar Spine <sup class="beta">beta</sup> | MRI | [rsna_2024_lumbar_spine.md](datasets/rsna_2024_lumbar_spine.md) |
| CT-RATE <sup class="beta">beta</sup> | CT | [ct_rate.md](datasets/ct_rate.md) |
| RAD-ChestCT <sup class="beta">beta</sup> | CT | [radchestct.md](datasets/radchestct.md) |
| TAIX-Ray | CXR | [taix_ray.md](datasets/taix_ray.md) |
| Shenzhen Hospital CXR | CXR | [shenzhen_cxr.md](datasets/shenzhen_cxr.md) |
| BRAX | CXR | [brax.md](datasets/brax.md) |
| RANZCR CLiP | CXR | [ranzcr_clip.md](datasets/ranzcr_clip.md) |
| RSNA 2022 Cervical Spine <sup class="beta">beta</sup> | CT | [rsna_2022_cervical_spine.md](datasets/rsna_2022_cervical_spine.md) |
| RSNA 2023 Abdominal Trauma <sup class="beta">beta</sup> | CT | [rsna_abdominal_trauma_2023.md](datasets/rsna_abdominal_trauma_2023.md) |
| OpenI IU CXR | CXR | [openi_cxr.md](datasets/openi_cxr.md) |
| Montgomery County CXR | CXR | [montgomery_cxr.md](datasets/montgomery_cxr.md) |
