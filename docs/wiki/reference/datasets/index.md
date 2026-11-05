# Datasets

<!-- claim: id=registry.n_dataset_keys tier=ci cmd="python -c 'import radharmony.dataset; from radharmony.registry import list_datasets; print(len(list_datasets()))'" expect="132" -->
RadHarmony has 132 dataset registry keys. Each row of the inventory is one dataset configuration. The
Registry keys column gives the keys for `resolve_dataset()`. `list_datasets()` in `radharmony.registry`
lists every key (import `radharmony.dataset` first). The Page column links to the page of the dataset
(constructor arguments, harmonizer notes, label columns).

When a row has split keys and one key without a split name (for example `chexpert`), use the split keys.
The key without a split name is the old class. A row that says "not for benchmarks" is an old class.
It mixes the official splits, or a folder or file that you give picks the split (D-09).

## Inventory

| Dataset | Modality | Dim | Labels | Outputs available | Registry keys | Page |
|---------|----------|-----|--------|-------------------|---------------|------|
| CheXpert | CXR | 2D | 14 pathologies | cls | `chexpert_train`, `chexpert_valid`, `chexpert` | [→](cxr-labels/chexpert.md) |
| CheXpert-Plus (Train) | CXR | 2D | 14 pathologies | cls, report | `chexpert_plus_train` | [→](cxr-labels/chexpert_plus.md) |
| CheXpert-Plus (Valid) | CXR | 2D | 14 pathologies | cls, report | `chexpert_plus_valid` | [→](cxr-labels/chexpert_plus.md) |
| CheXpert-Plus (all splits, not for benchmarks) | CXR | 2D | 14 pathologies | cls, report | `chexpert_plus` | [→](cxr-labels/chexpert_plus.md) |
| CheXphoto | CXR | 2D | 14 pathologies | cls | `chexphoto_train`, `chexphoto_valid`, `chexphoto` | [→](cxr-labels/chexphoto.md) |
| CheXlocalize (Test) | CXR | 2D | 14 pathologies | cls, mask | `chexlocalize` | [→](cxr-boxes-masks/chexlocalize.md) |
| CheXlocalize (Val) | CXR | 2D | 14 pathologies | cls, mask | `chexlocalize_val` | [→](cxr-boxes-masks/chexlocalize.md) |
| VQA-RAD (Train) | CXR/CT/MRI | 2D | n/a (VQA) | question, answer | `vqa_rad_train` | [→](vqa-text/vqa_rad.md) |
| VQA-RAD (Test) | CXR/CT/MRI | 2D | n/a (VQA) | question, answer | `vqa_rad_test` | [→](vqa-text/vqa_rad.md) |
| VQA-RAD (all splits, not for benchmarks) | CXR/CT/MRI | 2D | n/a (VQA) | question, answer | `vqa_rad` | [→](vqa-text/vqa_rad.md) |
| CheXinstruct VQA (Train) | CXR | 2D | n/a (VQA) | question, answer | `chexinstruct_vqa_train` | [→](vqa-text/chexinstruct_vqa.md) |
| CheXinstruct VQA (Val) | CXR | 2D | n/a (VQA) | question, answer | `chexinstruct_vqa_val` | [→](vqa-text/chexinstruct_vqa.md) |
| CheXinstruct VQA (Test) | CXR | 2D | n/a (VQA) | question, answer | `chexinstruct_vqa_test` | [→](vqa-text/chexinstruct_vqa.md) |
| CheXinstruct VQA (all splits, not for benchmarks) | CXR | 2D | n/a (VQA) | question, answer | `chexinstruct_vqa` | [→](vqa-text/chexinstruct_vqa.md) |
| MIMIC-CXR (DICOM, Train) | CXR | 2D | 14 pathologies | cls, report | `mimic_cxr_train` | [→](cxr-labels/mimic_cxr.md) |
| MIMIC-CXR (DICOM, Validate) | CXR | 2D | 14 pathologies | cls, report | `mimic_cxr_validate` | [→](cxr-labels/mimic_cxr.md) |
| MIMIC-CXR (DICOM, Test) | CXR | 2D | 14 pathologies | cls, report | `mimic_cxr_test` | [→](cxr-labels/mimic_cxr.md) |
| MIMIC-CXR (DICOM, all splits, not for benchmarks) | CXR | 2D | 14 pathologies | cls, report | `mimic_cxr` | [→](cxr-labels/mimic_cxr.md) |
| MIMIC-CXR-JPG (Train) | CXR | 2D | 14 pathologies | cls | `mimic_cxr_jpg_train` | [→](cxr-labels/mimic_cxr_jpg.md) |
| MIMIC-CXR-JPG (Validate) | CXR | 2D | 14 pathologies | cls | `mimic_cxr_jpg_validate` | [→](cxr-labels/mimic_cxr_jpg.md) |
| MIMIC-CXR-JPG (Test) | CXR | 2D | 14 pathologies | cls | `mimic_cxr_jpg_test` | [→](cxr-labels/mimic_cxr_jpg.md) |
| MIMIC-CXR-JPG (Test, labeler labels) | CXR | 2D | 14 pathologies | cls | `mimic_cxr_jpg_test_chexpert` | [→](cxr-labels/mimic_cxr_jpg.md) |
| MIMIC-CXR-JPG (all splits, not for benchmarks) | CXR | 2D | 14 pathologies | cls | `mimic_cxr_jpg` | [→](cxr-labels/mimic_cxr_jpg.md) |
| ChestX-ray14 | CXR | 2D | 15 pathologies | cls | `chestxray14_train`, `chestxray14_test`, `chestxray14` | [→](cxr-labels/chestxray14.md) |
| ChestX-Det10 | CXR | 2D | 10 pathologies (boxes) | cls, bbox | `chestxdet10_train`, `chestxdet10_test`, `chestxdet10` | [→](cxr-boxes-masks/chestxdet10.md) |
| NODE21 | CXR | 2D | nodule (boxes) | cls, bbox | `node21` | [→](cxr-boxes-masks/node21.md) |
| NODE21 (original images) | CXR | 2D | nodule (boxes) | cls, bbox | `node21_original` | [→](cxr-boxes-masks/node21.md) |
| ChestX-ray14 (BBox) | CXR | 2D | 15 pathologies | cls, bbox | `chestxray14_bbox` | [→](cxr-labels/chestxray14.md) |
| PadChest | CXR | 2D | 193 findings | cls, report (Spanish) | `padchest` | [→](cxr-labels/padchest.md) |
| PadChest-GR (Train / Validation / Test) | CXR | 2D | 154 findings (boxes) | cls, bbox, report | `padchest_gr_train`, `padchest_gr_validation`, `padchest_gr_test` | [→](cxr-boxes-masks/padchest_gr.md) |
| PadChest-GR (all splits, not for benchmarks) | CXR | 2D | 154 findings (boxes) | cls, bbox, report | `padchest_gr` | [→](cxr-boxes-masks/padchest_gr.md) |
| ReXGradient-160K (Train) | CXR | 2D | none (14 with `labels_csv_path`) | report, cls (with `labels_csv_path`) | `rexgradient_train`, `rexgradient` | [→](cxr-labels/rexgradient.md) |
| ReXGradient-160K (Valid) | CXR | 2D | none (14 with `labels_csv_path`) | report, cls (with `labels_csv_path`) | `rexgradient_valid`, `rexgradient` | [→](cxr-labels/rexgradient.md) |
| ReXGradient-160K (Test) | CXR | 2D | none (14 with `labels_csv_path`) | report, cls (with `labels_csv_path`) | `rexgradient_test`, `rexgradient` | [→](cxr-labels/rexgradient.md) |
| VinDr-CXR (Train) | CXR | 2D | 28 findings | cls, bbox | `vindr_cxr_train` | [→](cxr-boxes-masks/vindr_cxr.md) |
| VinDr-CXR (Test) | CXR | 2D | 28 findings | cls, bbox | `vindr_cxr_test` | [→](cxr-boxes-masks/vindr_cxr.md) |
| Chest ImaGenome (Gold) | CXR | 2D | per-box findings | bbox | `chest_imagenome_gold` | [→](cxr-boxes-masks/chest_imagenome.md) |
| Chest ImaGenome (Silver, Train / Valid / Test) | CXR | 2D | per-box findings | bbox | `chest_imagenome_silver_train`, `chest_imagenome_silver_valid`, `chest_imagenome_silver_test` | [→](cxr-boxes-masks/chest_imagenome.md) |
| Chest ImaGenome (Silver, all splits, not for benchmarks) | CXR | 2D | per-box findings | bbox | `chest_imagenome_silver` | [→](cxr-boxes-masks/chest_imagenome.md) |
| VinDr-PCXR (Train) | CXR | 2D | 15 conditions (pediatric) | cls, bbox | `vindr_pcxr_train` | [→](cxr-boxes-masks/vindr_pcxr.md) |
| VinDr-PCXR (Test) | CXR | 2D | 15 conditions (pediatric) | cls, bbox | `vindr_pcxr_test` | [→](cxr-boxes-masks/vindr_pcxr.md) |
| VinDr-PCXR (all splits, not for benchmarks) | CXR | 2D | 15 conditions (pediatric) | cls, bbox | `vindr_pcxr` | [→](cxr-boxes-masks/vindr_pcxr.md) |
| SIIM-ACR Pneumothorax | CXR | 2D | pneumothorax | cls, mask (test key: none) | `siim_acr_ptx_train`, `siim_acr_ptx_test`, `siim_acr_ptx` | [→](cxr-boxes-masks/siim_acr_ptx.md) |
| SIIM COVID-19 | CXR | 2D | 4 appearance classes | cls, bbox (test key: none) | `siim_covid19_train`, `siim_covid19_test`, `siim_covid19` | [→](cxr-boxes-masks/siim_covid19.md) |
| RSNA Pneumonia | CXR | 2D | 3 classes | cls, bbox | `rsna_pneumonia` | [→](cxr-boxes-masks/rsna_pneumonia.md) |
| RSNA Pneumonia (Kaggle) | CXR | 2D | 3 classes | cls, bbox (test key: none) | `rsna_pneumonia_kaggle_train`, `rsna_pneumonia_kaggle_test`, `rsna_pneumonia_kaggle` | [→](cxr-boxes-masks/rsna_pneumonia_kaggle.md) |
| RSNA PE Detection <sup class="beta">beta</sup> | CT | 3D | 13 PE labels | cls (test key: none) | `rsna_pe_detection_train`, `rsna_pe_detection_test`, `rsna_pe_detection` | [→](ct-mri-other/rsna_pe_detection.md) |
| RSNA Pediatric Bone Age <sup class="beta">beta</sup> | Radiograph | 2D | age (regression) | reg | `rsna_bone_age_train`, `rsna_bone_age_val`, `rsna_bone_age` | [→](ct-mri-other/rsna_bone_age.md) |
| RSNA Pediatric Bone Age (Test, no labels) <sup class="beta">beta</sup> | Radiograph | 2D | none | none | `rsna_bone_age_test` | [→](ct-mri-other/rsna_bone_age.md) |
| RSNA 2024 Lumbar Spine <sup class="beta">beta</sup> | MRI | 3D | 75 severity cols | cls, bbox (test key: none) | `rsna_2024_lumbar_spine_train`, `rsna_2024_lumbar_spine_test`, `rsna_2024_lumbar_spine` | [→](ct-mri-other/rsna_2024_lumbar_spine.md) |
| CT-RATE (Train / Valid) <sup class="beta">beta</sup> | CT | 3D | 18 findings | cls | `ct_rate_train`, `ct_rate_valid` | [→](ct-mri-other/ct_rate.md) |
| CT-RATE (folder picks the split, not for benchmarks) <sup class="beta">beta</sup> | CT | 3D | 18 findings | cls | `ct_rate` | [→](ct-mri-other/ct_rate.md) |
| RAD-ChestCT (Train / Valid / Test) <sup class="beta">beta</sup> | CT | 3D | 84 findings | cls, bbox | `radchestct_train`, `radchestct_valid`, `radchestct_test` | [→](ct-mri-other/radchestct.md) |
| RAD-ChestCT (label file picks the split, not for benchmarks) <sup class="beta">beta</sup> | CT | 3D | 84 findings | cls, bbox | `radchestct` | [→](ct-mri-other/radchestct.md) |
| Montgomery County CXR | CXR | 2D | TB classification | cls, mask, report | `montgomery_cxr` | [→](cxr-boxes-masks/montgomery_cxr.md) |
| Shenzhen Hospital CXR | CXR | 2D | tuberculosis (binary) | cls, mask, report | `shenzhen_cxr` | [→](cxr-boxes-masks/shenzhen_cxr.md) |
| QaTa-COV19 v2 (Train) | CXR | 2D | covid (constant 1) + infection masks | cls, mask | `qata_cov19_train`, `qata_cov19` | [→](cxr-boxes-masks/qata_cov19.md) |
| QaTa-COV19 v2 (Test) | CXR | 2D | covid (constant 1) + infection masks | cls, mask | `qata_cov19_test`, `qata_cov19` | [→](cxr-boxes-masks/qata_cov19.md) |
| MedSeg COVID-19 CT (Dataset 1) | CT | 2D slices | 3 lesion classes + any_lesion | cls, mask | `medseg_covid_ct` | [→](ct-mri-other/medseg_covid_ct.md) |
| MedSeg COVID-19 CT (Dataset 2) | CT | 2D slices | 3 lesion classes + any_lesion | cls, mask | `medseg_covid_ct` | [→](ct-mri-other/medseg_covid_ct.md) |
| MedSeg COVID-19 CT (Dataset 1, val_im, no labels) | CT | 2D slices | none | none | `medseg_covid_ct_val` | [→](ct-mri-other/medseg_covid_ct.md) |
| OpenI IU CXR | CXR | 2D | 8 findings (CheXpert-compat) | cls, report | `openi_cxr` | [→](vqa-text/openi_cxr.md) |
| OpenI IU CXR (DICOM) | CXR | 2D | 8 findings (CheXpert-compat) | cls, report | `openi_cxr` | [→](vqa-text/openi_cxr.md) |
| TAIX-Ray (512px, Train / Val / Test) | CXR | 2D | 8 findings (binary/ordinal) | cls | `taix_ray_512_train`, `taix_ray_512_val`, `taix_ray_512_test` | [→](cxr-labels/taix_ray.md) |
| TAIX-Ray (512px, all splits, not for benchmarks) | CXR | 2D | 8 findings (binary/ordinal) | cls | `taix_ray_512` | [→](cxr-labels/taix_ray.md) |
| TAIX-Ray (original, Train / Val / Test) | CXR | 2D | 8 findings (binary/ordinal) | cls | `taix_ray_train`, `taix_ray_val`, `taix_ray_test` | [→](cxr-labels/taix_ray.md) |
| TAIX-Ray (original, all splits, not for benchmarks) | CXR | 2D | 8 findings (binary/ordinal) | cls | `taix_ray` | [→](cxr-labels/taix_ray.md) |
| RSNA 2022 Cervical Spine <sup class="beta">beta</sup> | CT | 3D | 8 fracture labels | cls, mask (test key: none) | `rsna_2022_cervical_spine_train`, `rsna_2022_cervical_spine_test`, `rsna_2022_cervical_spine` | [→](ct-mri-other/rsna_2022_cervical_spine.md) |
| RSNA 2022 Cervical Spine (BBox) <sup class="beta">beta</sup> | CT | 3D | 8 fracture labels | cls, bbox | `rsna_2022_cervical_spine_bbox` | [→](ct-mri-other/rsna_2022_cervical_spine.md) |
| RSNA 2023 Abdominal Trauma <sup class="beta">beta</sup> | CT | 3D | 14 injury labels | cls (test key: none) | `rsna_abdominal_trauma_2023_train`, `rsna_abdominal_trauma_2023_test`, `rsna_abdominal_trauma_2023` | [→](ct-mri-other/rsna_abdominal_trauma_2023.md) |
| BRAX (DICOM) | CXR | 2D | 14 pathologies | cls | `brax` | [→](cxr-labels/brax.md) |
| BRAX (PNG) | CXR | 2D | 14 pathologies | cls | `brax_png` | [→](cxr-labels/brax.md) |
| RANZCR CLiP (Train) | CXR | 2D | 11 catheter/line labels | cls, mask | `ranzcr_clip_train` | [→](cxr-boxes-masks/ranzcr_clip.md) |
| RANZCR CLiP (Test, no labels) | CXR | 2D | none | none | `ranzcr_clip_test` | [→](cxr-boxes-masks/ranzcr_clip.md) |
| RANZCR CLiP (all splits, not for benchmarks) | CXR | 2D | 11 catheter/line labels | cls, mask | `ranzcr_clip` | [→](cxr-boxes-masks/ranzcr_clip.md) |
| GEMeX-VQA | CXR | 2D | n/a (VQA over MIMIC-CXR-JPG) | question, answer, bbox | `gemex_vqa` | [→](vqa-text/gemex_vqa.md) |
| MIMIC-Ext-CXR-QBA (Train / Validate / Test) | CXR | 2D | n/a (VQA over MIMIC-CXR) | question, answer | `mimic_ext_cxr_qba_train`, `mimic_ext_cxr_qba_validate`, `mimic_ext_cxr_qba_test` | [→](vqa-text/mimic_ext_cxr_qba.md) |
| MIMIC-Ext-CXR-QBA (all splits, not for benchmarks) | CXR | 2D | n/a (VQA over MIMIC-CXR) | question, answer | `mimic_ext_cxr_qba` | [→](vqa-text/mimic_ext_cxr_qba.md) |
| ROCO (Train / Validation / Test) | Multimodal | 2D | n/a (captioning) | question (empty), answer, keywords | `roco_train`, `roco_validation`, `roco_test` | [→](vqa-text/roco.md) |
| ROCO (all splits, not for benchmarks) | Multimodal | 2D | n/a (captioning) | question (empty), answer, keywords | `roco` | [→](vqa-text/roco.md) |
| ROCOv2 (Train / Validation / Test) | Radiology figures | 2D | n/a (captioning) | question (empty), answer | `rocov2_train`, `rocov2_validation`, `rocov2_test` | [→](vqa-text/roco.md) |
| ROCOv2 (all splits, not for benchmarks) | Radiology figures | 2D | n/a (captioning) | question (empty), answer | `rocov2` | [→](vqa-text/roco.md) |
| MedPMC | Medical figures (PubMed Central) | 2D | n/a (captioning) | question (empty), answer | `medpmc` | [→](vqa-text/medpmc.md) |
| EmoryCXR v2 (PNG / DICOM) | CXR | 2D | 14 pathologies | cls, report | `emory_cxr`, `emory_cxr_dicom` | [→](cxr-labels/emory_cxr.md) |
| Emory CHORUS (X-ray subset, Emory only) | CXR | 2D | none (images only) | img | `emory_chorus` | [→](cxr-labels/emory_chorus.md) |
| MS-CXR (Train / Val / Test) | CXR | 2D | 8 findings + phrase grounding | cls, bbox | `ms_cxr_train`, `ms_cxr_val`, `ms_cxr_test` | [→](cxr-boxes-masks/ms_cxr.md) |
| MS-CXR (all splits, not for benchmarks) | CXR | 2D | 8 findings + phrase grounding | cls, bbox | `ms_cxr` | [→](cxr-boxes-masks/ms_cxr.md) |
| MS-CXR-T | CXR | 2D | temporal progression (5 findings) | previous_img | `ms_cxr_t` | [→](cxr-labels/ms_cxr_t.md) |

## Finding a dataset's label columns

Every image dataset class has `LABEL_COLS` and `REG_COLS` as class attributes. The VQA and captioning
classes (VQA-RAD, CheXinstruct VQA, GEMeX-VQA, MIMIC-Ext-CXR-QBA, ROCO, ROCOv2 and MedPMC) do not have
them. An example for image classes:

```python
from radharmony.dataset import CheXpertDataset, RSNABoneAgeDataset

print(CheXpertDataset.LABEL_COLS)   # ['atelectasis', 'cardiomegaly', ...]
print(RSNABoneAgeDataset.REG_COLS)  # ['age_months']
```

The `cls` tensor in a data dict has one value for each entry in `LABEL_COLS`, in the order of
`sorted(LABEL_COLS)` (the plain Python sort). This order can differ from the order in the class definition.
See [Dataset API → Data dict keys](../api-datasets.md#data-dict-keys).

## Output flags

Each image dataset supports a subset of output flags. Passing a flag that the dataset does not support raises a warning and the key is absent from the data dict. The VQA and caption classes take none of these flags. They raise `TypeError` for an unknown argument (D-11).

| Flag | Data dict key | Description |
|------|---------------|-------------|
| `output_cls=True` | `"cls"` | Multi-label classification tensor |
| `output_mask=True` | `"mask"` | Segmentation mask tensor |
| `output_bbox=True` | `"bbox"`, `"bbox_labels"` | Float32 tensor `(N, 4)` (2-D) or `(N, 6)` (3-D) of fractional boxes, and a list of labels in snake case (see [Transforms → Boxes](../transforms.md#boxes)) |
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
| RSNA Bone Age | `boneage-training-dataset/` | `boneage-test-dataset/` (validation: `Bone Age Validation Set/`) |
| RANZCR CLiP | Kaggle root (images in `train/`) | Kaggle root (images in `test/`) |
| QaTa-COV19 v2 | `QaTa-COV19-v2/Train Set/Images/` | `QaTa-COV19-v2/Test Set/Images/` |
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
