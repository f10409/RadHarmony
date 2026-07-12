# NAS Dataset Paths

Quick-reference tables for every dataset's `base_image_dir` and annotation files on NAS3/NAS4.
All paths verified 2026-05-09.

---

## CheXpert

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS3/datasets/external/CheXpert-v1.0/train/` |
| `base_image_dir` (valid) | `/mnt/NAS3/datasets/external/CheXpert-v1.0/valid/` |
| `csv_path` (`train.csv`) | `/mnt/NAS3/datasets/external/CheXpert-v1.0/train.csv` |
| `csv_path` (`valid.csv`) | `/mnt/NAS3/datasets/external/CheXpert-v1.0/valid.csv` |

---

## CheXpert-Plus

| Role | Path |
|---|---|
| `base_image_dir` | `/mnt/NAS4/datasets/external/CheXpert_Plus/chexpertplus/DICOM/Uncompressed` |
| `csv_path` (`df_chexpert_plus_240401.csv`) | `/mnt/NAS4/datasets/external/CheXpert_Plus/chexpertplus/df_chexpert_plus_240401.csv` |
| `label_json_path` (`report_fixed.json`) | `/mnt/NAS4/datasets/external/CheXpert_Plus/chexpertplus/report_fixed.json` |

`impression_fixed.json` and `findings_fixed.json` also exist alongside `report_fixed.json`. CSVs and JSONs live 2 levels above `base_image_dir` and are auto-discoverable via `infer_path`.

---

## MIMIC-CXR (DICOM)

| Role | Path |
|---|---|
| `base_image_dir` | `/mnt/NAS4/datasets/external/MIMIC-CXR-V2-AWS/files/` |
| `csv_path` (`mimic-cxr-2.0.0-metadata.csv`) | `/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv` |
| `label_csv_path` (`mimic-cxr-2.0.0-chexpert.csv`) | `/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv` |
| `report_csv_path` (`cxr-study-list.csv.gz`) | `/mnt/NAS4/datasets/external/MIMIC-CXR-V2-AWS/cxr-study-list.csv.gz` |

Metadata and chexpert CSVs live in the JPG tree on NAS3, not in the DICOM tree. Pass them explicitly — auto-discovery may not find them across NAS mounts.

---

## MIMIC-CXR-JPG

| Role | Path |
|---|---|
| `base_image_dir` | `/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/files/` |
| `csv_path` (`mimic-cxr-2.0.0-metadata.csv`) | `/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-metadata.csv` |
| `label_csv_path` (`mimic-cxr-2.0.0-chexpert.csv`) | `/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-chexpert.csv` |
| `split_csv_path` (`mimic-cxr-2.0.0-split.csv`) | `/mnt/NAS3/datasets/external/MIMIC_CXR/physionet.org/files/mimic-cxr-jpg/2.0.0/mimic-cxr-2.0.0-split.csv` |

All CSVs live in the parent dir of `files/` and are auto-discoverable from `base_image_dir`.

---

## ChestX-ray14

| Role | Path |
|---|---|
| `base_image_dir` | `/mnt/NAS3/datasets/external/NIH_CXR/CXR14/` |
| `csv_path` (`Data_Entry_2017.csv`) | `/mnt/NAS3/datasets/external/NIH_CXR/CXR14/Data_Entry_2017.csv` |
| `bbox_csv_path` (`BBox_List_2017.csv`) | `/mnt/NAS3/datasets/external/NIH_CXR/CXR14/BBox_List_2017.csv` |

880 unique images have bounding boxes (984 boxes total across 8 classes). `ChestXray14BboxHarmonizer` requires `base_image_dir` to read image dimensions for bbox normalization.

---

## VinDr-CXR

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/train/` |
| `base_image_dir` (test) | `/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/test/` |
| `annotations_csv_path` (train) | `/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_train.csv` |
| `annotations_csv_path` (test) | `/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_test.csv` |
| image-level labels (train) | `/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_train.csv` |
| image-level labels (test) | `/mnt/NAS4/datasets/external/VinDr-CXR/physionet.org/files/vindr-cxr/1.0.0/annotations/image_labels_test.csv` |

---

## SIIM-ACR Pneumothorax

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS3/datasets/external/SIIM_ACR_Pneumothorax/dicom-images-train/` |
| `base_image_dir` (test) | `/mnt/NAS3/datasets/external/SIIM_ACR_Pneumothorax/dicom-images-test/` |
| `csv_path` (`train-rle.csv`) | `/mnt/NAS3/datasets/external/SIIM_ACR_Pneumothorax/train-rle.csv` |
| pre-decoded mask dir | `/mnt/NAS3/datasets/external/SIIM_ACR_Pneumothorax/masks` |

---

## SIIM COVID-19

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/siim-covid19-detection/train/` |
| `base_image_dir` (test) | `/mnt/NAS4/datasets/external/siim-covid19-detection/test/` |
| `csv_path` (`train_study_level.csv`) | `/mnt/NAS4/datasets/external/siim-covid19-detection/train_study_level.csv` |
| `image_csv_path` (`train_image_level.csv`) | `/mnt/NAS4/datasets/external/siim-covid19-detection/train_image_level.csv` |

CSVs live at the competition root (parent of `train/`), auto-discoverable via `infer_path`. `SIIMCOVID19TestDataset` does NOT accept `csv_path` or `image_csv_path`.

---

## RSNA Pneumonia — adjudicated JSON

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/rsna-pneumonia-detection-challenge/stage_2_train_images/` |
| `base_image_dir` (test) | `/mnt/NAS4/datasets/external/rsna-pneumonia-detection-challenge/stage_2_test_images/` |
| `csv_path` (adjudicated JSON) | **NOT ON NAS** — must be downloaded separately |

The harmonizer's `csv_path` points to `pneumonia-challenge-annotations-adjudicated-kaggle_2018.json`, not a CSV. NAS only has `stage_2_train_labels.csv` (used by the Kaggle variant).

---

## RSNA Pneumonia — Kaggle CSV

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/rsna-pneumonia-detection-challenge/stage_2_train_images/` |
| `base_image_dir` (test) | `/mnt/NAS4/datasets/external/rsna-pneumonia-detection-challenge/stage_2_test_images/` |
| `csv_path` (`stage_2_train_labels.csv`) | `/mnt/NAS4/datasets/external/rsna-pneumonia-detection-challenge/stage_2_train_labels.csv` |

---

## RSNA PE Detection

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/rsna_pe_dataset/train/` |
| `base_image_dir` (test) | `/mnt/NAS4/datasets/external/rsna_pe_dataset/test/` |
| `csv_path` (`train.csv`) | `/mnt/NAS4/datasets/external/rsna_pe_dataset/train.csv` |
| test CSV (`test.csv`) | `/mnt/NAS4/datasets/external/rsna_pe_dataset/test.csv` |

---

## RSNA Pediatric Bone Age

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/rsna-boneage/boneage-training-dataset/` |
| `base_image_dir` (val) | `/mnt/NAS4/datasets/external/rsna-boneage/Bone Age Validation Set/` |
| `csv_path` (train, `train.csv`) | `/mnt/NAS4/datasets/external/rsna-boneage/train.csv` |
| `csv_path` (val, `Validation Dataset.csv`) | `/mnt/NAS4/datasets/external/rsna-boneage/Bone Age Validation Set/Validation Dataset.csv` |

---

## RSNA 2022 Cervical Spine

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_images/` |
| `base_image_dir` (test) | `/mnt/NAS4/datasets/external/rsna-2022-cervical-spine-fracture-detection/test_images/` |
| `csv_path` (`train.csv`) | `/mnt/NAS4/datasets/external/rsna-2022-cervical-spine-fracture-detection/train.csv` |
| `bbox_csv_path` (`train_bounding_boxes.csv`) | `/mnt/NAS4/datasets/external/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv` |

---

## RSNA 2023 Abdominal Trauma

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/rsna-2023-abdominal-trauma-detection/train_images/` |
| `base_image_dir` (test) | `/mnt/NAS4/datasets/external/rsna-2023-abdominal-trauma-detection/test_images/` |
| `csv_path` (`train_2024.csv`) | `/mnt/NAS4/datasets/external/rsna-2023-abdominal-trauma-detection/train_2024.csv` |
| `series_meta_csv_path` (`train_series_meta.csv`) | `/mnt/NAS4/datasets/external/rsna-2023-abdominal-trauma-detection/train_series_meta.csv` |
| test series meta (`test_series_meta.csv`) | `/mnt/NAS4/datasets/external/rsna-2023-abdominal-trauma-detection/test_series_meta.csv` |

Only one dataset class: `RSNAAbdominalTrauma2023Dataset` (no separate Train/Test class).

---

## RSNA 2024 Lumbar Spine

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train_images/` |
| `base_image_dir` (test) | `/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/test_images/` |
| `csv_path` (`train.csv`) | `/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train.csv` |
| `series_desc_csv_path` (`train_series_descriptions.csv`) | `/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv` |
| `label_coord_csv_path` (`train_label_coordinates.csv`) | `/mnt/NAS4/datasets/external/rsna-2024-lumbar-spine-degenerative-classification/train_label_coordinates.csv` |

---

## CT-RATE

| Role | Path |
|---|---|
| `base_image_dir` (train) | `/mnt/NAS4/datasets/external/CT-RATE/dataset/train_fixed/` |
| `base_image_dir` (valid) | `/mnt/NAS4/datasets/external/CT-RATE/dataset/valid_fixed/` |
| `csv_path` (train, `train_predicted_labels.csv`) | `/mnt/NAS4/datasets/external/CT-RATE/dataset/tables/train_predicted_labels.csv` |
| `csv_path` (valid, `valid_predicted_labels.csv`) | `/mnt/NAS4/datasets/external/CT-RATE/dataset/tables/valid_predicted_labels.csv` |
| `view_position_csv_path` (train, `train_metadata.csv`) | `/mnt/NAS4/datasets/external/CT-RATE/dataset/tables/train_metadata.csv` |
| `view_position_csv_path` (valid, `validation_metadata.csv`) | `/mnt/NAS4/datasets/external/CT-RATE/dataset/tables/validation_metadata.csv` |

CSVs live in `tables/` (sibling of `train_fixed/` and `valid_fixed/`), auto-discoverable via `infer_path`.

---

## RAD-ChestCT

| Role | Path |
|---|---|
| `image_base_dir` | `/mnt/NAS4/datasets/external/RAD-ChestCT/images/` |
| `csv_path` (`CT_Scan_Metadata_Complete_35747.csv`) | `/mnt/NAS4/datasets/external/RAD-ChestCT/tables/CT_Scan_Metadata_Complete_35747.csv` |
| `label_csv_path` (train) | `/mnt/NAS4/datasets/external/RAD-ChestCT/tables/imgtrain_Abnormality_and_Location_Labels.csv` |
| `label_csv_path` (valid) | `/mnt/NAS4/datasets/external/RAD-ChestCT/tables/imgvalid_Abnormality_and_Location_Labels.csv` |
| `label_csv_path` (test) | `/mnt/NAS4/datasets/external/RAD-ChestCT/tables/imgtest_Abnormality_and_Location_Labels.csv` |

The constructor parameter is `image_base_dir=` (not `base_image_dir=`) — legacy name. CSVs live in `tables/` sibling dir, auto-discoverable from `images/`.

---

## TAIX-Ray

| Role | Path |
|---|---|
| `base_image_dir` (512 px) | `/mnt/NAS4/datasets/external/TAIX-Ray/data_512/images/` |
| `base_image_dir` (original) | `/mnt/NAS4/datasets/external/TAIX-Ray/data_original/` |
| `csv_path` (512 px, `annotation.csv`) | `/mnt/NAS4/datasets/external/TAIX-Ray/data_512/annotation.csv` |
| `csv_path` (original, `annotation.csv`) | `/mnt/NAS4/datasets/external/TAIX-Ray/data_original/annotation.csv` |

## Montgomery County CXR

| Role | Path |
|---|---|
| `base_image_dir` (`CXR_png/`) | `/mnt/NAS4/datasets/external/Montgomery-CXR/MontgomerySet/CXR_png/` |
| Sibling clinical readings | `/mnt/NAS4/datasets/external/Montgomery-CXR/MontgomerySet/ClinicalReadings/` |
| Sibling manual masks | `/mnt/NAS4/datasets/external/Montgomery-CXR/MontgomerySet/ManualMask/` |

## Shenzhen Hospital CXR

| Role | Path |
|---|---|
| `base_image_dir` (`CXR_png/`) | `/mnt/NAS4/datasets/external/Shenzhen-Hospital-CXR-Set/CXR_png/` |
| Sibling clinical readings | `/mnt/NAS4/datasets/external/Shenzhen-Hospital-CXR-Set/ClinicalReadings/` |
| Sibling TB finding annotations | `/mnt/NAS4/datasets/external/Shenzhen-Hospital-CXR-Set/Annotations-2/` |
