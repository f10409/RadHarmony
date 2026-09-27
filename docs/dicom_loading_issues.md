# DICOM loading issues in the 2-D pipeline

Findings from checking how the 2-D pipeline (`_base_load_2d` in
[radharmony/dataset/transforms.py](../radharmony/dataset/transforms.py)) reads DICOM files.
The pipeline reads DICOMs with MONAI's `ITKReader` (ITK / GDCM), then applies the DICOM window
(`_apply_voi_lut`), then scales intensities to [-1, 1].

## Summary

| # | Issue | Affected | Status |
|---|---|---|---|
| 1 | "MONOCHROME1 images are served inverted" | nothing | Not a bug |
| 2 | Window applied to flipped MONOCHROME1 pixels gives wrong contrast | CheXpert-Plus, 10 of 131 sampled MONOCHROME1 files with window tags | Fixed |
| 3 | JPEG 2000 files are flipped with a different constant | VinDr-CXR JPEG 2000 files (would turn black) | Fixed |
| 4 | Files with no `HighBit` tag load as a blank image | CheXpert-Plus, about 5% of files | Fixed |
| 5 | `.dicom` files skip the window step | VinDr-CXR | Harmless, no change |
| 6 | Leftover partial downloads named `.azDownload-*.dcm` | CheXpert-Plus, 18 of 5,436 sampled files | Open (data cleanup) |

**After updating:** `PersistentDataset` caches (`cache_dir`) and embedding caches built from
CheXpert-Plus DICOMs still hold the old images. Delete them so they are rebuilt with the fixes.

## Background

A DICOM stores each pixel as a number. There are two conventions:

- **MONOCHROME2:** big number = bright (bone bright, air dark). Most models expect this.
- **MONOCHROME1:** the opposite, big number = dark.

**ITK flips MONOCHROME1 images while reading them**, so they come out in MONOCHROME2 style.
It computes `slope * (M - raw) + intercept`, where `M = 2**BitsStored - 1`.
Example, 12-bit file: a stored value of 95 becomes `4095 - 95 = 4000`.

## 1. "MONOCHROME1 images are served inverted": not a bug

`transforms.py:420` has the MONOCHROME1 flip commented out
(`# ... func=_convert_monochrome1_to_2`, commit `3788e79`). It was reported that this makes
MONOCHROME1 images (541 of 3,000 VinDr-CXR test images) come out inverted.

This is not the case, because ITK already flips them. Checked on VinDr-CXR test: ITK pixels vs raw
pydicom pixels have correlation **+1.0** for MONOCHROME2 files and **-1.0** for MONOCHROME1 files.
Turning line 420 back on would flip MONOCHROME1 images a second time and make them wrong.
**Leave it commented out.**

## 2. Wrong contrast from the window step (fixed)

**Cause.** The window tags (`WindowCenter` / `WindowWidth`) describe the numbers as stored in the
file, before ITK's flip. The old `_apply_voi_lut` applied them to the numbers after ITK's flip.

Example: the header says "show stored values 0 to 1000". After ITK's flip those pixels are
3095 to 4095, but the window still selects 0 to 1000, so it picks the wrong pixels.

When the window is centered in the pixel range, flipping does not move it, so the result is still
correct. Only off-center windows go wrong.

**Impact (CheXpert-Plus, 5,418 sampled files).** 131 are MONOCHROME1 with window tags; **10 came out
wrong**. The worst file was mostly washed out to white (mean abs diff 0.600 on the [-1, 1] scale).
In SIIM-ACR, SIIM-COVID, and RSNA Pneumonia no sampled files have window tags, so the step does nothing there.

**Fix** in `_apply_voi_lut` ([radharmony/dataset/base.py](../radharmony/dataset/base.py)):
undo ITK's flip, apply the window, flip back. It only runs for MONOCHROME1 files with window tags;
all other files are handled exactly as before.

```python
flip = getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1" and (
    "WindowCenter" in ds or "VOILUTSequence" in ds
)
if flip:
    s = float(ds.get("RescaleSlope", 1) or 1)
    b = float(ds.get("RescaleIntercept", 0) or 0)
    m = 2 ** int(ds.BitsStored) - 1
    if (arr.max() - b) / s > m:          # JPEG 2000, see issue 3
        m = 2 ** int(ds.BitsAllocated) - 1
    arr = (s * m + 2 * b) - arr          # undo ITK's flip
arr = apply_voi_lut(arr, ds).astype("float32")
if flip:
    arr = arr.max() + arr.min() - arr    # flip back
```

**Verified.** 131 of 131 MONOCHROME1 files now match a pydicom-only reference image; 10 MONOCHROME2
control files are unchanged; `pytest tests/` passes (90 passed, 3 skipped).

## 3. JPEG 2000 flip constant (fixed, part of the same change)

**Cause.** For JPEG 2000 files (transfer syntax `1.2.840.10008.1.2.4.90`), ITK flips around
`2**BitsAllocated - 1` (65535) instead of `2**BitsStored - 1` (4095). A 12-bit value of 95 becomes
65440, not 4000.

**Impact.** With the old code, the window (for example center 2047, width 4095) sees only values
61440 to 65535, so every pixel saturates and the image turns **fully black**. This happens in the
VinDr-CXR JPEG 2000 files when the window step runs on them. Today the step does not run on VinDr
(see issue 5), so no current output is affected.

**Fix.** The two cases are easy to tell apart from ITK's output: flipped 12-bit JPEG 2000 values
are at least 61440, far above 4095. The code above picks 65535 in that case.
Verified on 20 of 20 VinDr JPEG 2000 files: after undoing the flip, values land in 0 to 4095.

## 4. Missing `HighBit` loads a blank image (fixed)

**Cause.** Each pixel is stored in a 16-bit box (`BitsAllocated = 16`), but often only 12 or 15 bits
are used (`BitsStored`). `HighBit` says where the used bits sit in the box, and is almost always
`BitsStored - 1`. Some CheXpert-Plus files are missing it. pydicom assumes the usual value and reads
them fine; ITK reads the wrong bits and every pixel comes out as 1, so the image is blank.

**Impact.** Sampled 16 DICOM datasets (400 files each). **Only CheXpert-Plus is affected**:
about 5% of its files have no `HighBit` (366 of 5,418 in a larger sample), and 259 of those load
blank. All other datasets had `HighBit` in every sampled file: MIMIC-CXR, VinDr-CXR train/test,
SIIM-ACR, SIIM-COVID, RSNA Pneumonia, BRAX, Emory CXR (DICOM), Emory CHORUS, and the 3-D RSNA PE,
Cervical Spine, Abdominal Trauma, and Lumbar Spine sets. VinDr-PCXR and OpenI could not be checked
(no read permission).

**Fix.** ITK stays the reader. A new step, `_fix_missing_highbit`
([radharmony/dataset/base.py](../radharmony/dataset/base.py)), runs in `_base_load_2d` right before
the window step. If a DICOM has no `HighBit`, it re-reads that one file with pydicom and writes the
pixels in the same form ITK uses (rescale applied, MONOCHROME1 as `slope * (M - raw) + intercept`),
so the window step and everything after it are unchanged. ITK still provides the image shape and
metadata. Files that have `HighBit` are not touched. Cost: one extra header read per DICOM image.

```python
if "HighBit" in pydicom.dcmread(path, stop_before_pixels=True):
    return x                                   # normal file: keep ITK pixels
ds = pydicom.dcmread(path)
arr = ds.pixel_array.astype("float32")
if ds.PhotometricInterpretation == "MONOCHROME1":
    arr = (2 ** int(ds.BitsStored) - 1) - arr  # same flip as ITK
arr = slope * arr + intercept
```

This was chosen over skipping these files in the CheXpert-Plus harmonizer, which would need a header
read of every file up front (about 220k files on the NAS) and would drop about 5% of images.

**Verified.** All 366 sampled CheXpert-Plus files with no `HighBit` now match the pydicom reference
(241 MONOCHROME1, 3 MONOCHROME1 with window tags, 122 MONOCHROME2). Files with `HighBit` are unchanged
by the new step (58 of 58 across CheXpert-Plus, VinDr-CXR, SIIM-COVID). `pytest tests/`: 90 passed, 3 skipped.
## 5. `.dicom` files skip the window step (no change needed)

`_apply_voi_lut` only runs on paths ending in `.dcm`. VinDr-CXR files end in `.dicom`, so they are
never windowed (2,941 of 3,000 test files have window tags). This is harmless: VinDr windows cover
the full pixel range, and the output with or without windowing differs by 0.000 to 0.001 (mean abs
diff on [-1, 1]). If this check is ever widened to `.dicom`, the issue 3 fix is required first.

## 6. `.azDownload-*` leftover files (open)

The CheXpert-Plus DICOM folder contains leftover partial downloads named like
`.azDownload-<id>-view1_frontal.dcm` (18 of 5,436 sampled). They are not valid DICOM. They only
matter if a harmonizer ever lists them; they should be deleted from the NAS copy.

## Reproduce

[notebooks/datasets/dicom_monochrome_voi_verify.ipynb](../notebooks/datasets/dicom_monochrome_voi_verify.ipynb)
reruns every check above and plots the affected images (old vs fixed vs reference). The outputs
contain patient images: clear them before committing.
