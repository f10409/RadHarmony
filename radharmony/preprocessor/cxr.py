"""CXR (2D) preprocessor for chest X-rays and other 2D radiological images."""

import SimpleITK as sitk

from .base import RadiologicPreprocessor


class CXRPreprocessor(RadiologicPreprocessor):
    """Preprocessor for chest X-rays and other 2D radiological images.

    Rescales intensity to [-1, 1] using the image's actual min/max, optionally
    crops to the largest foreground region, then resizes isotropically.

    Supports DICOM (.dcm), PNG, and JPEG inputs.

    Args:
        target_long_side: Desired pixel count for the longest side (default: 112).
        interpolation: SimpleITK interpolation method (default: sitkLinear).
    """

    def __init__(
        self,
        target_long_side: int = 112,
        interpolation=sitk.sitkLinear,
    ):
        super().__init__(target_long_side=target_long_side, interpolation=interpolation)

    def __call__(self, image_path: str) -> sitk.Image:
        """Run the 2D preprocessing pipeline on one image.

        Args:
            image_path: Path to a 2D image (DICOM, PNG, JPEG).

        Returns:
            Preprocessed SimpleITK image.
        """
        # sitk/GDCM applies the Modality LUT (RescaleSlope/Intercept) automatically.
        image = sitk.ReadImage(image_path)

        if image_path.lower().endswith(".dcm"):
            image = self._fix_monochrome(image)
            image = self._apply_voi_lut(image, image_path)

        image = self._to_grayscale_float(image)
        image = self._rescale(image)
        # image = self._crop_2d(image)
        image = self._resize_isotropic(image)

        return image

    def _apply_voi_lut(self, image: sitk.Image, image_path: str) -> sitk.Image:
        """Apply VOI LUT from DICOM metadata (WindowCenter/Width or LUT sequence).

        Reads only the DICOM header (no pixel data re-read) and uses pydicom's
        apply_voi_lut, which handles both linear windowing and lookup tables.
        """
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        ds  = pydicom.dcmread(image_path, stop_before_pixels=True)
        arr = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32))
        arr = apply_voi_lut(arr, ds).astype("float32")
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(image)
        return out

    def _rescale(self, image: sitk.Image) -> sitk.Image:
        """Rescale intensity to [-1, 1] using the image's actual min/max."""
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        return sitk.IntensityWindowing(
            image,
            windowMinimum=stats.GetMinimum(),
            windowMaximum=stats.GetMaximum(),
            outputMinimum=-1.0,
            outputMaximum=1.0,
        )
