# Intent: preprocessors and utils

**Design decisions that apply** (`docs/wiki/design.md`). Test each one, and read its known gaps:
- D-13: CXR pixel values
- D-16: backbone features
- D-27: readers
- D-35: `infer_path`
- D-42: channel count
- D-43: a resample keeps the image centered
- Any other entry in the design.md section "Dataset classes and transforms" that applies to this part.

**Other test steps:**

1. **CXR preprocessor** (`radharmony/preprocessor/cxr.py`, `base.py`): the DICOM rescale slope and
   intercept and the VOI LUT or window are applied as documented. The output dtype and range are as
   documented. The aspect ratio is kept when the docs say so, padding is symmetric, and 12-bit and 16-bit
   images work. (Inversion: D-13.)
2. **CT preprocessor** (`ct.py`): HU = raw * slope + intercept. The window (center and width) clips to the
   documented range. Slice order follows the patient axis (ImagePositionPatient), not the file name
   order. Spacing and resampling are as documented.
3. **Utils** (`radharmony/utils/data_utils.py`, `mask_utils.py`, `infer.py`):
   - `infer_path` finds the documented file and never picks a wrong file with a similar name.
   - The RLE encode and decode round trip is exact, with the documented column-major or row-major order.
   - The mask helpers keep the shape.
4. **Evaluator transforms and wrappers** (`radharmony/evaluator/transforms.py`, `wrappers.py`):
   - The resize and normalization that a backbone needs are applied once, not twice.
   - Grayscale goes to 3 channels when the backbone needs RGB.
   - Wrappers return features with the documented shape (D-16).

**How to test:** use small synthetic arrays and a few real DICOMs, for example from SIIM-ACR (`.dcm`), RSNA PE
and CT-RATE. Find them with the data paths file or in the data folders of your task (`references/parts.md`, "Data and environments").
