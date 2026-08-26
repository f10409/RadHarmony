# Datasets

RadHarmony supports 46 dataset configurations across 36 underlying datasets. Click a dataset name for the full page (constructor args, harmonizer notes, label columns).

## Inventory

| Dataset | Modality | Dim | Labels | Outputs available | Page |
|---------|----------|-----|--------|-------------------|------|
| CheXpert | CXR | 2D | 14 pathologies | cls | [→](datasets/chexpert.md) |
| CheXpert-Plus | CXR | 2D | 14 pathologies | cls, report | [→](datasets/chexpert_plus.md) |
| CheXlocalize | CXR | 2D | 14 pathologies | cls, mask | [→](datasets/chexlocalize.md) |
| VQA-RAD | CXR/CT/MRI | 2D | n/a (VQA) | question, answer | [→](datasets/vqa_rad.md) |
| MIMIC-CXR (DICOM) | CXR | 2D | 14 pathologies | cls, report | [→](datasets/mimic_cxr.md) |
| MIMIC-CXR-JPG | CXR | 2D | 14 pathologies | cls | [→](datasets/mimic_cxr_jpg.md) |
| MIMIC-CXR-JPG (Test) | CXR | 2D | 14 pathologies | cls | [→](datasets/mimic_cxr_jpg.md) |
| ChestX-ray14 | CXR | 2D | 15 pathologies | cls | [→](datasets/chestxray14.md) |
| ChestX-ray14 (BBox) | CXR | 2D | 15 pathologies | cls, bbox | [→](datasets/chestxray14.md) |
| PadChest | CXR | 2D | 193 findings | cls, report (Spanish) | [→](datasets/padchest.md) |
| ReXGradient-160K (Train) | CXR | 2D | none | report | [→](datasets/rexgradient.md) |
| ReXGradient-160K (Valid) | CXR | 2D | none | report | [→](datasets/rexgradient.md) |
| ReXGradient-160K (Test) | CXR | 2D | none | report | [→](datasets/rexgradient.md) |
| VinDr-CXR (Train) | CXR | 2D | 28 findings | cls, bbox | [→](datasets/vindr_cxr.md) |
| VinDr-CXR (Test) | CXR | 2D | 28 findings | cls, bbox | [→](datasets/vindr_cxr.md) |
| Chest ImaGenome (Gold) | CXR | 2D | per-box findings | bbox | [→](datasets/chest_imagenome.md) |
| Chest ImaGenome (Silver) | CXR | 2D | per-box findings | bbox | [→](datasets/chest_imagenome.md) |
| VinDr-PCXR | CXR | 2D | 15 conditions (pediatric) | cls, bbox | [→](datasets/vindr_pcxr.md) |
| SIIM-ACR Pneumothorax | CXR | 2D | pneumothorax | cls, mask | [→](datasets/siim_acr_ptx.md) |
| SIIM COVID-19 | CXR | 2D | 4 appearance classes | cls, bbox | [→](datasets/siim_covid19.md) |
| RSNA Pneumonia | CXR | 2D | 3 classes | cls, bbox | [→](datasets/rsna_pneumonia.md) |
| RSNA Pneumonia (Kaggle) | CXR | 2D | 3 classes | cls, bbox | [→](datasets/rsna_pneumonia_kaggle.md) |
| RSNA PE Detection <sup class="beta">beta</sup> | CT | 3D | 13 PE labels | cls | [→](datasets/rsna_pe_detection.md) |
| RSNA Pediatric Bone Age <sup class="beta">beta</sup> | Radiograph | 2D | age (regression) | reg | [→](datasets/rsna_bone_age.md) |
| RSNA 2024 Lumbar Spine <sup class="beta">beta</sup> | MRI | 3D | 75 severity cols | cls, bbox | [→](datasets/rsna_2024_lumbar_spine.md) |
| CT-RATE <sup class="beta">beta</sup> | CT | 3D | 18 findings | cls | [→](datasets/ct_rate.md) |
| RAD-ChestCT <sup class="beta">beta</sup> | CT | 3D | 84 findings | cls, mask, report, bbox | [→](datasets/radchestct.md) |
| Montgomery County CXR | CXR | 2D | TB classification | cls, mask | [→](datasets/montgomery_cxr.md) |
| Shenzhen Hospital CXR | CXR | 2D | tuberculosis (binary) | cls, mask, report | [→](datasets/shenzhen_cxr.md) |
| OpenI IU CXR | CXR | 2D | 8 findings (CheXpert-compat) | cls, report | [→](datasets/openi_cxr.md) |
| OpenI IU CXR (DICOM) | CXR | 2D | 8 findings (CheXpert-compat) | cls, report | [→](datasets/openi_cxr.md) |
| TAIX-Ray (512px) | CXR | 2D | 8 findings (binary/ordinal) | cls | [→](datasets/taix_ray.md) |
| TAIX-Ray (original) | CXR | 2D | 8 findings (binary/ordinal) | cls | [→](datasets/taix_ray.md) |
| RSNA 2022 Cervical Spine <sup class="beta">beta</sup> | CT | 3D | 8 fracture labels | cls, mask | [→](datasets/rsna_2022_cervical_spine.md) |
| RSNA 2022 Cervical Spine (BBox) <sup class="beta">beta</sup> | CT | 3D | 8 fracture labels | cls, bbox | [→](datasets/rsna_2022_cervical_spine.md) |
| RSNA 2023 Abdominal Trauma <sup class="beta">beta</sup> | CT | 3D | 14 injury labels | cls | [→](datasets/rsna_abdominal_trauma_2023.md) |
| BRAX (DICOM) | CXR | 2D | 14 pathologies | cls | [→](datasets/brax.md) |
| BRAX (PNG) | CXR | 2D | 14 pathologies | cls | [→](datasets/brax.md) |
| RANZCR CLiP | CXR | 2D | 11 catheter/line labels | cls, mask | [→](datasets/ranzcr_clip.md) |
| GEMeX-VQA | CXR | 2D | n/a (VQA over MIMIC-CXR-JPG) | question, answer, bbox | [→](datasets/gemex_vqa.md) |
| MIMIC-Ext-CXR-QBA | CXR | 2D | n/a (VQA over MIMIC-CXR) | question, answer | [→](datasets/mimic_ext_cxr_qba.md) |
| ROCO | Multimodal | 2D | n/a (captioning) | question (empty), answer, keywords | [→](datasets/roco.md) |
| EmoryCXR v2 | CXR | 2D | 14 pathologies | cls, report | [→](datasets/emory_cxr.md) |
| Emory CHORUS (X-ray subset) | CXR | 2D | none (image-only) | img | [→](datasets/emory_chorus.md) |
| MS-CXR | CXR | 2D | 8 findings + phrase grounding | cls, bbox | [→](datasets/ms_cxr.md) |
| MS-CXR-T | CXR | 2D | temporal progression (5 findings) | previous_img | [→](datasets/ms_cxr_t.md) |

## Finding a dataset's label columns

Every dataset class exposes `LABEL_COLS` and `REG_COLS` as class attributes:

```python
from radharmony.dataset import CheXpertDataset, RSNABoneAgeDataset

print(CheXpertDataset.LABEL_COLS)   # ['atelectasis', 'cardiomegaly', ...]
print(RSNABoneAgeDataset.REG_COLS)  # ['age_months']
```

The `cls` tensor in a data dict has one value per entry in `LABEL_COLS`, in **sorted (alphabetical) order** — not necessarily the order they appear in the class definition.

## Output flags

Each dataset supports a subset of output flags. Passing a flag that the dataset does not support raises a warning and the key is absent from the data dict.

| Flag | Data dict key | Description |
|------|---------------|-------------|
| `output_cls=True` | `"cls"` | Multi-label classification tensor |
| `output_mask=True` | `"mask"` | Segmentation mask tensor |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | List of `[dim0_min, dim0_max, dim1_min, dim1_max]` fractional boxes |
| `output_report=True` | `"report"` | Free-text radiology report string |
| `output_reg=True` | `"reg"` | Regression target tensor |

## `base_image_dir` convention

`base_image_dir` is the **deepest common ancestor directory** shared by all
images in the dataset — not the competition root. `image_path` values are
relative to it and may include subdirectory components (e.g.
`patient_id/study_id/image.dcm` for datasets with patient/study folder
structure). For each dataset, the wiki page above documents the exact expected
layout and an example path.

For datasets with separate train/test splits, each split takes its own image
directory:

| Dataset | Train base | Test base |
|---|---|---|
| CheXpert | `train/` | `valid/` |
| ChestX-ray14 | CXR14 root (`images_001/` … `images_012/` siblings) | same |
| VinDr-CXR | `train/` | `test/` |
| SIIM-ACR-PTX | `dicom-images-train/` | `dicom-images-test/` |
| SIIM COVID-19 | `train/` | `test/` |
| RSNA Pneumonia (Kaggle) | `stage_2_train_images/` | `stage_2_test_images/` |
| RSNA PE Detection | `train/` | `test/` |
| RSNA Cervical Spine | `train_images/` | `test_images/` |
| RSNA Abdominal Trauma | `train_images/` | `test_images/` |
| RSNA Lumbar Spine | `train_images/` | `test_images/` |
| RSNA Bone Age | `boneage-training-dataset/` | `Bone Age Validation Set/` |
| RANZCR CLiP | `train/` | `test/` |
| ReXGradient-160K | `deid_png/` (train split) | `deid_png/` (test split — JSON differs per split, image tree shared) |

**Structural exceptions** — some datasets have images split across multiple
sibling subdirs inside `base_image_dir`, where `image_path` carries the
subdir name. Examples:

- **MIMIC-CXR (DICOM) and MIMIC-CXR-JPG**: `base_image_dir` is the `files/`
  directory; images are grouped under `p10/`, `p11/`, … patient-group
  subdirectories, so `image_path` takes the form
  `p{group}/p{patient_id}/s{study_id}/{image}.dcm` (or `.jpg`).
- **ChestX-ray14**: `base_image_dir` is the CXR14 root; images are split
  across its `images_001/` … `images_012/` sibling children.
- **PadChest**: `base_image_dir` is the `images/` directory; images are
  split across its `0/`, `1/`, …, `50/`, `54/` sibling children (slots
  51/52/53 don't exist in the official distribution).
- **RSNA Bone Age (val)**: `base_image_dir` is `Bone Age Validation Set/`;
  images are split across its `boneage-validation-dataset-1/` and `-2/`
  sibling children. (Train has no exception — its `base_image_dir` is the
  flat image directory.)

When adding a new dataset, follow the `base_image_dir` convention: no
hardcoded subdirectory prefix in `image_path` that could be folded into
`base_image_dir` instead.
