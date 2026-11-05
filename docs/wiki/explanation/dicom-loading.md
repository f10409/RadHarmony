# DICOM loading

A DICOM file can store its pixels in several ways. If the loader reads them the wrong way, the image comes
out inverted, too dark, washed out or blank, and no error tells you. This page explains how the 2-D
pipeline reads a DICOM file, and why each step is there. It also lists the problems that we found and
fixed in 2026. The rule is D-13: every chest X-ray that a dataset class gives has bright bone.

## The 2-D load steps

`_base_load_2d` in `radharmony/dataset/transforms.py` gives the load steps of every 2-D pipeline:

| Step | What it does |
|---|---|
| `LoadImageD(reader="ITKReader")` | Reads the file with ITK. ITK uses GDCM for DICOM files |
| `_drop_dicom_axis`, `TransposeD` | Gives the image the shape `(C, H, W)` |
| `_fix_missing_highbit` | Reads the pixels again with pydicom when the file has no `HighBit` tag |
| `_apply_voi_lut` | Applies the window of a MONOCHROME file (`WindowCenter` and `WindowWidth`, or a VOI LUT) |
| `_SetChannelsD` | Sets the channel count of the dataset (D-42) |
| `ScaleIntensityRangePercentilesD` | Scales the 0.5 and 99.5 percentiles to [-1, 1], and clips (D-14) |

The 2 DICOM steps (`_fix_missing_highbit` and `_apply_voi_lut`) are in `radharmony/dataset/base.py`. A
backbone recipe with `reader="monai"` runs the same 2 steps ([Backbones](../reference/backbones/index.md)).
`CXRPreprocessor` in `radharmony/preprocessor/cxr.py` has its own copy, with the same rules.

`is_dicom` in `radharmony/utils/dicom.py` decides if a file is a DICOM file. It looks for the Photometric
Interpretation tag (0028,0004), with any file name. It skips `.mha`, `.nrrd` and NIfTI files, because a
copy of a DICOM file in these formats keeps the tag. The 2 DICOM steps do nothing for a file that is not
DICOM.

## MONOCHROME1 and MONOCHROME2

A DICOM file stores each pixel as a number. There are 2 conventions:

- MONOCHROME2: a large number is bright. Bone is bright and air is dark. Most models expect this.
- MONOCHROME1: a large number is dark.

ITK inverts a MONOCHROME1 image when it reads the file. It gives `s * (M - raw) + b`, where `s` and `b`
are `RescaleSlope` and `RescaleIntercept`, and `M = 2**BitsStored - 1`. Example: in a 12-bit file, the
stored value 95 becomes `4095 - 95 = 4000`. `M` is the largest plus the smallest stored
value (`stored_value_sum` in `radharmony/utils/dicom.py`). For signed pixel values (`PixelRepresentation`
1), this sum is -1. Example for 16 bits: 32767 + (-32768) = -1.

So the pipeline has no MONOCHROME1 inversion step. A second inversion makes the image wrong. The old pipeline
had such a step in a comment, so it never ran. We removed it on 2026-10-29.

<!-- claim: id=dicom_loading.vindr_test.images tier=ci cmd="python scripts/evalkit/dataset_stat.py vindr-cxr-test vindr_cxr_test.images" expect="3,000" -->
A review said that MONOCHROME1 images (541 of the 3,000 VinDr-CXR test images) come out inverted. This is
not true. On VinDr-CXR test, the ITK pixels and the raw pydicom pixels have a correlation of +1.0 for
MONOCHROME2 files and -1.0 for MONOCHROME1 files. A check of 6 VinDr-CXR test files on 2026-11-04 gave the
same result. The pipeline output had a correlation of +1.0 with a pydicom reference for both kinds.

## The window of a MONOCHROME1 file

The window tags (`WindowCenter` and `WindowWidth`) refer to the stored values, before the ITK inversion.
The old `_apply_voi_lut` applied them to the values after the inversion.

Example: the header says "show the stored values 0 to 1000". After the ITK inversion, these pixels are
3095 to 4095, but the window still takes 0 to 1000. So it takes the wrong pixels. A window in the center
of the range does not move when the image is inverted. So only a window that is not in the center gives a
wrong image.

In a sample of CheXpert Plus files, 131 were MONOCHROME1 with window tags, and 10 of them came out wrong.
The worst file was almost all white (a mean absolute difference of 0.600 on the [-1, 1] scale). In
SIIM-ACR, SIIM-COVID and RSNA Pneumonia, no sampled file has window tags, so the window step does nothing
there.

The fix: for a MONOCHROME1 file with window tags, `_apply_voi_lut` undoes the ITK inversion, applies the
window, and inverts the image again. It does not change any other file. After the fix, 131 of 131
MONOCHROME1 files matched a reference image made with pydicom only, and 10 MONOCHROME2 files did not
change.

## JPEG 2000 files

For a JPEG 2000 file (transfer syntax `1.2.840.10008.1.2.4.90`), ITK inverts around
`2**BitsAllocated - 1` (65535), not `2**BitsStored - 1` (4095). So a 12-bit value of 95 becomes 65440, not
4000. With the wrong `M`, the window (for example center 2047, width 4095) sees only the values 61440 to
65535. Every pixel is then out of the window, and the image is all black. This happened in the VinDr-CXR
JPEG 2000 files.

It is easy to see which case a file is in: an inverted 12-bit JPEG 2000 value is 61440 or more, far above 4095.
When a value is above `M`, `_apply_voi_lut` uses `2**BitsAllocated - 1`. In 20 of 20 VinDr JPEG 2000
files, the values were in 0 to 4095 after the step undid the inversion.

## Files with no `HighBit` tag

A pixel is stored in 16 bits (`BitsAllocated` 16), but often only 12 or 15 of the bits are used
(`BitsStored`). `HighBit` says where the used bits are. It is almost always `BitsStored - 1`. Some
CheXpert Plus files have no `HighBit` tag. pydicom then uses `BitsStored - 1` and reads the file
correctly. ITK reads the wrong bits, and every pixel is 1, so the image is blank.

We tried to sample 400 files from each of 16 DICOM datasets. VinDr-CXR train and VinDr-CXR test count as 2.
We could not read 2 of them. Only CheXpert Plus has the problem. About 5% of its
files have no `HighBit` (366 files in a larger sample), and 259 of these loaded blank. These datasets had
`HighBit` in every sampled file:

- 2-D: MIMIC-CXR, VinDr-CXR (train and test), SIIM-ACR, SIIM-COVID, RSNA Pneumonia, BRAX, Emory CXR and Emory
  CHORUS.
- 3-D: RSNA PE, RSNA Cervical Spine, RSNA Abdominal Trauma and RSNA Lumbar Spine.

We could not check VinDr-PCXR and OpenI, because we had no read access to them at that time.

The fix: ITK stays the reader. `_fix_missing_highbit` reads the header of each DICOM file. When the file
has no `HighBit` tag, it reads the pixels again with pydicom. It writes them in the ITK form (rescale
applied, MONOCHROME1 inverted with `M`), so the next steps do not change.

`_fix_missing_highbit` does nothing for a file with `HighBit`, and for a color file (`SamplesPerPixel`
not 1). When pydicom cannot decode the file, it warns and keeps the ITK pixels. The cost is one more
header read for each DICOM image.

The other choice was to drop these files in the CheXpert Plus harmonizer. That needs a header read of
every file before the first use, and it drops about 5% of the images. After the fix, all 366 sampled
files with no `HighBit` matched the pydicom reference. 58 of 58 files with `HighBit` did not change.

## Any file name

At first, `_apply_voi_lut` ran only on a path that ends in `.dcm`. The VinDr-CXR and VinDr-PCXR files end
in `.dicom`, so they had no window. About 97% of the VinDr-CXR files have window tags, and no file has a
VOI LUT sequence. On 60 VinDr-CXR test files, the window changed 11 images by more than 0.01 (mean
absolute difference on [-1, 1]), and one image by 0.36.

The fix first added `.dicom` to the file name check. Now both DICOM steps find a DICOM file by its
Photometric Interpretation tag, with any file name (see [The 2-D load steps](#the-2-d-load-steps)). This
fix needs the JPEG 2000 fix, because many VinDr-CXR files are JPEG 2000.

## Partial downloads in CheXpert Plus

The CheXpert Plus DICOM folder that we downloaded has partial downloads with names like
`.azDownload-<id>-view1_frontal.dcm` (18 in a sample of CheXpert Plus files). They are not valid DICOM
files. The harmonizer makes each `image_path` from the CSV file, so it never reads them. They are still
there on 2026-11-04.

## After an update

The fixes above were made before 2026-11-02. Since 2026-11-02, the key of a `cache_dir` folder holds the code
of each cached step (D-15). So a cache folder from before that date is not read again, and you do not need to
delete it for these fixes. Delete it only to free the disk. An embedding cache from before 2026-11-03 raises
`ValueError` (D-41). Delete it and run again.

A fix in a helper that a step calls, for example `stored_value_sum`, does not change the cache key. After
such a fix, delete the cache yourself ([Cache the samples](../how-to/cache.md)).

## Check it yourself

The notebook
[`notebooks/datasets/dicom_monochrome_voi_verify.ipynb`](https://github.com/f10409/RadHarmony/blob/main/notebooks/datasets/dicom_monochrome_voi_verify.ipynb)
runs the checks on this page again. It shows the affected images: the old load, the fixed load and the pydicom
reference. Its output has patient images, so clear the output before a commit.
