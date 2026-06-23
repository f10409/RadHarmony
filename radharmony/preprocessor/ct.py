"""CT (3D) preprocessor for CT volumes and other 3D radiological images."""

import SimpleITK as sitk

from .base import RadiologicPreprocessor


class CTPreprocessor(RadiologicPreprocessor):
    """Preprocessor for CT volumes and other 3D radiological images.

    Applies HU windowing, axial foreground cropping, isotropic resizing,
    and DICOM orientation correction.

    Supports any 3D format readable by SimpleITK (.nii, .nii.gz, .nrrd,
    .mha, .mhd, etc.).

    Args:
        target_long_side: Desired voxel count for the longest axial dimension (default: 112).
        interpolation: SimpleITK interpolation method (default: sitkLinear).
        orientation: DICOM orientation string applied to output (default: "LIP").
        window_min: Lower bound for HU windowing (default: -1000).
        window_max: Upper bound for HU windowing (default: 1000).
    """

    def __init__(
        self,
        target_long_side: int = 112,
        interpolation=sitk.sitkLinear,
        orientation: str = "LIP",
        window_min: float = -1000,
        window_max: float = 1000,
    ):
        super().__init__(target_long_side=target_long_side, interpolation=interpolation)
        self.orientation = orientation
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image_path: str) -> sitk.Image:
        """Run the 3D preprocessing pipeline on one volume.

        Args:
            image_path: Path to a 3D volume (NIfTI, NRRD, MHA, etc.).

        Returns:
            Preprocessed SimpleITK image.
        """
        image = sitk.ReadImage(image_path)
        image = self._to_grayscale_float(image)
        image = self._window(image)
        image = self._crop_3d(image)
        image = self._resize_isotropic(image)
        try:
            image = sitk.DICOMOrient(image, self.orientation)
        except RuntimeError:
            pass  # Missing/degenerate direction cosines — skip reorientation

        return image

    def _window(self, image: sitk.Image) -> sitk.Image:
        """Clip HU range and rescale to [-1, 1]."""
        return sitk.IntensityWindowing(
            image,
            windowMinimum=self.window_min,
            windowMaximum=self.window_max,
            outputMinimum=-1.0,
            outputMaximum=1.0,
        )


# ==========================================
# QUICK TEST  (python -m src.preprocessor.ct)
# ==========================================

if __name__ == "__main__":
    import sys
    import numpy as np

    IMAGE_PATH = (
        "/path/to/CT-RATE/dataset/train_fixed"
        "/train_1/train_1_a/train_1_a_1.nii.gz"
    )

    if len(sys.argv) > 1:
        IMAGE_PATH = sys.argv[1]

    print(f"Input : {IMAGE_PATH}")

    preprocessor = CTPreprocessor(
        target_long_side=112,
        window_min=-1000,
        window_max=1000,
        orientation="LIP",
    )

    # --- step-by-step inspection ---
    raw = sitk.ReadImage(IMAGE_PATH)
    s0 = preprocessor._to_grayscale_float(raw)
    s1 = preprocessor._window(s0)
    s2 = preprocessor._crop_3d(s1)
    s3 = preprocessor._resize_isotropic(s2)
    try:
        s4 = sitk.DICOMOrient(s3, preprocessor.orientation)
    except RuntimeError:
        s4 = s3
        print("  [warn] DICOMOrient skipped — degenerate direction cosines")

    for label, img in [("raw", s0), ("windowed", s1), ("cropped", s2), ("resized", s3), ("oriented", s4)]:
        arr = sitk.GetArrayFromImage(img)
        print(
            f"  {label:10s}  size={img.GetSize()}  "
            f"spacing={[round(s, 3) for s in img.GetSpacing()]}  "
            f"val=[{arr.min():.3f}, {arr.max():.3f}]"
        )

    # --- full pipeline ---
    out = preprocessor(IMAGE_PATH)
    arr = sitk.GetArrayFromImage(out)
    assert out.GetDimension() == 3, "Output must be 3D"
    assert arr.min() >= -1.0 and arr.max() <= 1.0, "Values must be in [-1, 1]"
    assert max(out.GetSize()[:2]) == 112, f"Longest axial side must be 112, got {out.GetSize()}"
    print("Full pipeline: passed.")

    # --- run_batch ---
    import os
    import tempfile
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = os.path.join(tmp_dir, "output")

        df_batch = pd.DataFrame({"Image Path": [IMAGE_PATH]})

        preprocessor.run_batch(df=df_batch, base_output_dir=out_dir, num_workers=1)

        outputs = os.listdir(out_dir)
        assert len(outputs) == 1, f"Expected 1 output file, got {outputs}"

        result = sitk.ReadImage(os.path.join(out_dir, outputs[0]))
        result_arr = sitk.GetArrayFromImage(result)
        assert result.GetDimension() == 3, "Batch output must be 3D"
        assert result_arr.min() >= -1.0 and result_arr.max() <= 1.0, "Batch output values must be in [-1, 1]"
        print(f"run_batch : passed.  output={outputs[0]}  size={result.GetSize()}")

    print("\nAll assertions passed.")
