"""CT (3D) preprocessor for CT volumes and other 3D radiological images."""

import SimpleITK as sitk

from .base import RadiologicPreprocessor


def _valid_orientation(code) -> bool:
    """Return True if ``code`` has exactly one letter of each pair L/R, A/P and S/I."""
    return (
        isinstance(code, str)
        and len(code) == 3
        and all(sum(c in pair for c in code) == 1 for pair in ("LR", "AP", "SI"))
    )


class CTPreprocessor(RadiologicPreprocessor):
    """Preprocessor for CT volumes and other 3D radiological images.

    Applies HU windowing, axial foreground cropping, isotropic resizing,
    and DICOM orientation correction. The resize also resamples the depth to
    the same spacing as the axial plane. Example: a CT-RATE volume has
    512 x 512 x 303 voxels (x, y, z). The crop and the resize give 112 x 76 x 84
    voxels, so the depth is 84 slices. The LIP orientation then gives 112 x 84 x 76.

    Supports the 3D formats that SimpleITK reads, for example .nii, .nii.gz,
    .nrrd, .mha and .mhd. It reads one file. It does not read a folder of
    DICOM slices.

    Args:
        target_long_side: Desired voxel count for the longest axial dimension (default: 112).
        interpolation: SimpleITK interpolation method (default: sitkLinear).
        orientation: DICOM orientation code of the output, 3 capital letters with one
            letter from each pair L/R, A/P and S/I (default: "LIP").
            Other strings raise ``ValueError``.
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
        if not _valid_orientation(orientation):
            raise ValueError(
                f"orientation must be 3 capital letters, one of L/R, one of A/P and one of S/I, got {orientation!r}"
            )
        self.orientation = orientation
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image_path: str) -> sitk.Image:
        """Run the 3D preprocessing pipeline on one volume.

        Args:
            image_path: Path to a 3D volume (NIfTI, NRRD or MHA).

        Returns:
            Preprocessed SimpleITK image.
        """
        image = sitk.ReadImage(image_path)
        image = self._to_grayscale_float(image)
        image = self._window(image)
        # The crop and the resize take index axes 0 and 1 as the axial plane.
        # So put L-R, A-P, S-I on the index axes first (V-P-26).
        try:
            image = sitk.DICOMOrient(image, "LPS")
        except RuntimeError:
            pass  # Missing or degenerate direction cosines: skip reorientation
        image = self._crop_3d(image)
        image = self._resize_isotropic(image)
        try:
            image = sitk.DICOMOrient(image, self.orientation)
        except RuntimeError:
            pass  # Missing or degenerate direction cosines: skip reorientation

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


# Quick test: python -m radharmony.preprocessor.ct <volume>

if __name__ == "__main__":
    import sys

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

    raw = sitk.ReadImage(IMAGE_PATH)
    s0 = preprocessor._to_grayscale_float(raw)
    s1 = preprocessor._window(s0)
    try:
        s1 = sitk.DICOMOrient(s1, "LPS")
    except RuntimeError:
        pass
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

    out = preprocessor(IMAGE_PATH)
    arr = sitk.GetArrayFromImage(out)
    assert out.GetDimension() == 3, "Output must be 3D"
    assert arr.min() >= -1.0 and arr.max() <= 1.0, "Values must be in [-1, 1]"
    # LIP gives (L-R, S-I, A-P), so the axial sides are index axes 0 and 2.
    assert max(out.GetSize()[0], out.GetSize()[2]) == 112, f"Longest axial side must be 112, got {out.GetSize()}"
    print("Full pipeline: passed.")

    import os
    import tempfile
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_dir = os.path.join(tmp_dir, "output")

        df_batch = pd.DataFrame({"image_path": [IMAGE_PATH]})

        preprocessor.run_batch(df=df_batch, base_output_dir=out_dir, num_workers=1)

        # The output keeps the folders of image_path below out_dir.
        outputs = [os.path.join(d, f) for d, _, files in os.walk(out_dir) for f in files]
        assert len(outputs) == 1, f"Expected 1 output file, got {outputs}"

        result = sitk.ReadImage(outputs[0])
        result_arr = sitk.GetArrayFromImage(result)
        assert result.GetDimension() == 3, "Batch output must be 3D"
        assert result_arr.min() >= -1.0 and result_arr.max() <= 1.0, "Batch output values must be in [-1, 1]"
        print(f"run_batch : passed.  output={os.path.relpath(outputs[0], out_dir)}  size={result.GetSize()}")

    print("\nAll assertions passed.")
