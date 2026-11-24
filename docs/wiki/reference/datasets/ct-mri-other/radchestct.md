# RAD-ChestCT

<!-- claim: id=radchestct.train.images tier=ci cmd="python scripts/evalkit/dataset_stat.py radchestct radchestct_train.images" expect="2,286" -->
<!-- claim: id=radchestct.train.patients tier=ci cmd="python scripts/evalkit/dataset_stat.py radchestct radchestct_train.patients" expect="2,114" -->
| | |
|---|---|
| Registry keys | `radchestct_train`, `radchestct_valid`, `radchestct_test` |
| Task and outputs | Classification, 84 labels (`cls`). One lung box for each volume (`bbox`) |
| Images in the table | train 2,286 volumes (2,114 patients), valid 984 volumes (616 patients), test 360 volumes (342 patients) |
| Modality | Chest CT, 3-D, NumPy `.npz`. One row is one scan, which is one volume |
| License | Duke University owns the data. CC BY-NC-ND 4.0, or a separate license from Duke. Ask for access on Zenodo before the download |
| Download | [Get the data: RAD-ChestCT](../../../how-to/get-the-data.md#rad-chestct) |

<!-- claim: id=radchestct.meta.rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source radchestct_train csv_path rows" expect="35,747" -->
<!-- claim: id=radchestct.all.images tier=ci cmd="python scripts/evalkit/dataset_stat.py radchestct radchestct_train.images+radchestct_valid.images+radchestct_test.images" expect="3,630" -->
RAD-ChestCT is a chest CT dataset from Duke University
([Draelos et al., Medical Image Analysis 2021](https://doi.org/10.1016/j.media.2020.101857)). A text tool
read each report and marked 84 findings at 52 locations. The metadata file lists 35,747 scans. The Zenodo
record has only the first release, about 10% of the scans. It has 3,630 volumes with labels, in 3 label
CSVs. The record does not say how to get the other scans.

## Quick use

`base_image_dir` is the `images/` folder, with one `<NoteAcc_DEID>.npz` file for each volume. The class
finds the CSV files in `base_image_dir` or in the folder above it, also in their `tables/` folders.

```
RAD-ChestCT/
  tables/CT_Scan_Metadata_Complete_35747.csv
  tables/Extrema_35747.csv
  tables/imgtrain_Abnormality_and_Location_Labels.csv
  tables/imgvalid_Abnormality_and_Location_Labels.csv
  tables/imgtest_Abnormality_and_Location_Labels.csv
  images/trn00022.npz
```

```python
import torch
from radharmony.dataset import RadChestCTTrainDataset, RadChestCTTestDataset

train = RadChestCTTrainDataset(
    base_image_dir="/data/RAD-ChestCT/images/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
)
train_ds, val_ds = train.get_datasets(n_splits=5)  # 1/5 of the patients go to val
sample = train_ds[0]
# sample["img"]: Tensor(1, 112, 112, 112), 1 channel, then (D, H, W)
# sample["cls"]: Tensor(84,)
# sample["bbox"]: Tensor(N, 6), the lung box. N is 1, or 0 for a volume with no box
# sample["bbox_labels"]: a list of N names, ['lung'] or []

test_ds = RadChestCTTestDataset(
    base_image_dir="/data/RAD-ChestCT/images/",
    output_cls=True,
    output_bbox=True,
    dtype=torch.float32,
).get_datasets()
```

The image steps: the `.npz` arrays have no orientation data, but they are already in the `IPL` orientation
and in `(D, H, W)` order. So the class passes `base_transpose=False`, and the transform does not turn the
volume (D-12). The transform clips the values to the fixed window `(-1000, 400)` (D-14). Then it scales the
0.5 and 99.5 percentiles to -1 and 1. It resizes the longest side to 112, and pads the volume with -1 to
112 x 112 x 112 (D x H x W).

The class has no argument for the window, so pass your own `transform` for another window. The files we
checked hold int16 HU values from -1000 to 1000.

The first run writes a MONAI cache to `./cache`. `cache_dir=None` gives no cache.

## How it is harmonized

- Source: one label CSV, `img<split>_Abnormality_and_Location_Labels.csv`. One row is one volume.
- IDs: the `NoteAcc_DEID` `tst29112` gives `study_id` `tst29112` and `image_path` `tst29112.npz`. The
  metadata CSV holds its `MRN_DEID` `mrn16164`, which gives `patient_id` `mrn16164`. `image_path` is
  relative to `base_image_dir` (D-04). One study is one volume.
- Labels (D-05): a source column is `<finding>*<location>`, for example `nodule*left_upper`. A finding is 1
  when one of its locations is 1. Else it is 0. The source has no uncertain or blank label.
- View (D-01): the metadata CSV has `orig_orientation`, but the harmonizer does not read it. So the
  harmonized table has no `view_position`.
- Age and sex: the source has no age or sex, so the harmonized table has no `age_years` or `sex` (D-06,
  D-26).
- Split: a split class needs the label CSV of its split. A label CSV of another split raises `ValueError`.
- Dropped rows: the harmonizer drops a volume with no `.npz` file in `base_image_dir`. Every volume of the
  label CSVs has a file.

## Labels

The 84 label columns, in the order of `cls`:

| Column | Meaning |
|---|---|
| `air_trapping` | Air trapping |
| `airspace_disease` | Airspace disease |
| `aneurysm` | An aneurysm |
| `arthritis` | Arthritis |
| `aspiration` | Aspiration |
| `atelectasis` | Part of a lung is collapsed |
| `atherosclerosis` | Atherosclerosis |
| `bandlike_or_linear` | A band-like or linear opacity |
| `breast_implant` | A breast implant |
| `breast_surgery` | Signs of breast surgery |
| `bronchial_wall_thickening` | The bronchial walls are thick |
| `bronchiectasis` | The bronchi are wide |
| `bronchiolectasis` | The bronchioles are wide |
| `bronchiolitis` | Bronchiolitis |
| `bronchitis` | Bronchitis |
| `cabg` | Coronary artery bypass graft |
| `calcification` | A calcification |
| `cancer` | Cancer |
| `cardiomegaly` | The heart is large |
| `catheter_or_port` | A catheter or a port |
| `cavitation` | A cavity |
| `chest_tube` | A chest tube |
| `clip` | A surgical clip |
| `congestion` | Vascular congestion |
| `consolidation` | Airspace consolidation |
| `coronary_artery_disease` | Coronary artery disease |
| `cyst` | A cyst |
| `debris` | Debris in an airway |
| `deformity` | A deformity |
| `density` | A dense area |
| `dilation_or_ectasia` | A wide vessel or duct |
| `distention` | A distended organ |
| `emphysema` | Emphysema |
| `fibrosis` | Fibrosis |
| `fracture` | A fracture |
| `gi_tube` | A gastrointestinal tube |
| `granuloma` | A granuloma |
| `groundglass` | Ground-glass opacity |
| `hardware` | Surgical hardware |
| `heart_failure` | Heart failure |
| `heart_valve_replacement` | A replaced heart valve |
| `hemothorax` | Blood in the pleural space |
| `hernia` | A hernia |
| `honeycombing` | Honeycombing |
| `infection` | An infection |
| `infiltrate` | An infiltrate |
| `inflammation` | Inflammation |
| `interstitial_lung_disease` | Interstitial lung disease |
| `lesion` | A lesion |
| `lucency` | A lucent area |
| `lung_resection` | Part of a lung was removed |
| `lymphadenopathy` | Large lymph nodes |
| `mass` | A mass |
| `mucous_plugging` | Mucus plugs in an airway |
| `nodule` | A nodule |
| `nodulegr1cm` | A nodule larger than 1 cm |
| `opacity` | An opacity |
| `other_path` | Another finding |
| `pacemaker_or_defib` | A pacemaker or a defibrillator |
| `pericardial_effusion` | Pericardial effusion |
| `pericardial_thickening` | The pericardium is thick |
| `plaque` | A plaque |
| `pleural_effusion` | Pleural effusion |
| `pleural_thickening` | The pleura is thick |
| `pneumonia` | Pneumonia |
| `pneumonitis` | Pneumonitis |
| `pneumothorax` | Pneumothorax |
| `postsurgical` | Changes after surgery |
| `pulmonary_edema` | Pulmonary edema |
| `reticulation` | A reticular pattern |
| `scarring` | Scarring |
| `scattered_calc` | Scattered calcifications |
| `scattered_nod` | Scattered nodules |
| `secretion` | Secretions |
| `septal_thickening` | The septa are thick |
| `soft_tissue` | A soft tissue finding |
| `staple` | A surgical staple |
| `stent` | A stent |
| `sternotomy` | Signs of a sternotomy |
| `suture` | A suture |
| `tracheal_tube` | A tracheal tube |
| `transplant` | A transplant |
| `tree_in_bud` | Tree-in-bud pattern |
| `tuberculosis` | Tuberculosis |

## Boxes

<!-- claim: id=radchestct.extrema.rows tier=nas cmd="python scripts/evalkit/dataset_stat.py --source radchestct_train bbox_csv_path rows" expect="35,747" -->
<!-- claim: id=radchestct.train.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py radchestct radchestct_train.boxes" expect="2,272" -->
<!-- claim: id=radchestct.train.images.boxes tier=ci cmd="python scripts/evalkit/dataset_stat.py radchestct radchestct_train.images" expect="2,286" -->
- Source: `Extrema_35747.csv` has one lung box for each of its 35,747 volumes. 6 columns hold the voxel
  index of the first and last lung voxel on each axis: `sup_axis0min` and `inf_axis0max`, `ant_axis1min`
  and `pos_axis1max`, `rig_axis2min` and `lef_axis2max`. The columns `shape0`, `shape1` and `shape2` hold
  the size of the volume.
- The harmonizer divides the 2 axis 0 values by `shape0`, the axis 1 values by `shape1`, and the axis 2
  values by `shape2`. The volume is `(D, H, W)` in `IPL` order, so axis 0 goes from superior to inferior.
- The box is `[dim0_min, dim0_max, dim1_min, dim1_max, dim2_min, dim2_max]`, in fractions of the volume
  size (D-07).
- Example: `tst29112` has the values 33, 362, 145, 363, 68 and 408, and the size `(406, 470, 470)` (D, H, W). It
  gives `bbox` `[0.0812807881773399, 0.8916256157635468, 0.30851063829787234, 0.7723404255319148,
  0.14468085106382977, 0.8680851063829788]` and `bbox_labels` `["lung"]`.
- `bbox_labels`: `["lung"]` for each box (D-07).
- Readers (D-36): the source has one box for each volume, so the harmonizer does not group boxes.
- No box: a volume with an `inf` value in the source gets `bbox` `[]` and `bbox_labels` `[]`. Box counts:
  train 2,272 boxes on 2,286 volumes, valid 983 on 984, and test 360 on 360.
- The sample: `bbox` is a float32 tensor `(N, 6)`, in fractions of the returned volume (D-28). `bbox_labels`
  is a list of N names. Example: `tst29112` gives `(1, 6)` and `['lung']`. The pad to a cube moves the box
  on axis 0, so its first 2 values are 0.1329 and 0.8347. A volume with no box gives `(0, 6)` and `[]`.
- Dropped boxes: the harmonizer clips each box to the volume. It drops a box with no area after the clip,
  and warns (D-07). The source files have no such box.

## Splits

<!-- claim: id=radchestct.splits.train tier=ci cmd="python scripts/evalkit/dataset_stat.py radchestct radchestct_train.images" expect="2,286" -->
| Class | Registry key | Images | Source |
|---|---|---|---|
| `RadChestCTTrainDataset` | `radchestct_train` | 2,286 | `imgtrain_Abnormality_and_Location_Labels.csv` |
| `RadChestCTValidDataset` | `radchestct_valid` | 984 | `imgvalid_Abnormality_and_Location_Labels.csv` |
| `RadChestCTTestDataset` | `radchestct_test` | 360 | `imgtest_Abnormality_and_Location_Labels.csv`. Use it as the test split |
| `RadChestCTDataset` | `radchestct` | | The old key. It reads the label CSV of `label_csv_path`. Without it, it reads the train labels. Use the split classes for a benchmark (D-09) |

`get_datasets(n_splits=5)` makes a train and val split from a split class. It groups by patient (D-10). No
patient is in 2 splits. The 3 split classes read the same `images/` folder, and the label CSV picks the
volumes.

## Arguments

The arguments are the same for `RadChestCTTrainDataset`, `RadChestCTValidDataset` and
`RadChestCTTestDataset`.

<!-- args: radharmony.dataset.RadChestCTTrainDataset -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `base_image_dir` | `str` | Yes | `None` | The `images/` folder with the `.npz` files. Not needed with `harmonizer_path`, which holds it |
| `csv_path` | `str` | No | `None` | `CT_Scan_Metadata_Complete_35747.csv`. Without it, the class looks in `base_image_dir` or in the folder above it, also in their `tables/` folders |
| `label_csv_path` | `str` | No | `None` | The label CSV of the split. Without it, the class looks in `base_image_dir` or in the folder above it, also in their `tables/` folders |
| `bbox_csv_path` | `str` | No | `None` | `Extrema_35747.csv`. Without it, the class looks in `base_image_dir` or in the folder above it, also in their `tables/` folders. With no file, the harmonized table has no box |

### Shared arguments (inherited from BaseRadiologicalDataset)

| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `transform` | MONAI transform | No | `None` | The image steps. `None` gives `RadiologyTransform3D` with `img_size=112`, `hu_window=(-1000, 400)` and `base_transpose=False` |
| `output_cls` | `bool` | No | `False` | Add the `cls` tensor to the sample |
| `output_mask` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_report` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11). The Zenodo record has no reports |
| `output_reg` | `bool` | No | `False` | Not supported. The class warns and adds no key (D-11) |
| `output_bbox` | `bool` | No | `False` | Add `bbox` and `bbox_labels` to the sample |
| `cache_dir` | `str` | No | `"./cache"` | The MONAI cache folder. `None` gives no cache |
| `dtype` | `torch.dtype` | No | `torch.bfloat16` | The type of `img`. The labels stay float32 |
| `harmonized_df` | `pd.DataFrame` | No | `None` | A harmonized table to use |
| `harmonizer` | harmonizer | No | `None` | A harmonizer to use |
| `harmonizer_path` | `str` | No | `None` | A saved harmonizer. See [Save and reuse a harmonized table](../../../how-to/save-and-reuse.md) |

## Harmonizer

```python
from radharmony.harmonizer import RadChestCTTrainHarmonizer

h = RadChestCTTrainHarmonizer(
    csv_path="/data/RAD-ChestCT/tables/CT_Scan_Metadata_Complete_35747.csv",
    label_csv_path="/data/RAD-ChestCT/tables/imgtrain_Abnormality_and_Location_Labels.csv",
    bbox_csv_path="/data/RAD-ChestCT/tables/Extrema_35747.csv",
    base_image_dir="/data/RAD-ChestCT/images/",
)
df = h.harmonize()
print(df.columns.tolist())
# ['patient_id', 'study_id', 'image_path', 'bbox', 'bbox_labels', 'air_trapping', ...]
```

`RadChestCTValidHarmonizer` and `RadChestCTTestHarmonizer` take the same arguments, with the label CSV of
their split.

<!-- args: radharmony.harmonizer.RadChestCTTrainHarmonizer -->
| Argument | Type | Required | Default | Description |
|---|---|---|---|---|
| `csv_path` | `str` | Yes | | `CT_Scan_Metadata_Complete_35747.csv`. It holds `MRN_DEID`, the patient ID |
| `label_csv_path` | `str` | Yes | | The label CSV of the split |
| `image_base_dir` | `str` | No | `None` | The old name of `base_image_dir`. It still works. With 2 different folders in the 2 names, the harmonizer raises `ValueError` |
| `bbox_csv_path` | `str` | No | `None` | `Extrema_35747.csv`. Without it, the harmonized table has no box |
| `base_image_dir` | `str` | No | `None` | The `images/` folder. With it, the harmonizer drops a volume with no `.npz` file. `verify_images()` uses it (D-04) |

## Known data problems

- Boxes with `inf`: `Extrema_35747.csv` has `inf` and `-inf` values for 173 volumes. 14 train volumes and 1
  valid volume are among them. Example: `val26894`.
