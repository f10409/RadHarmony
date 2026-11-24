"""CXR (2D) preprocessor for chest X-rays and other 2D radiological images."""

import warnings

import numpy as np
import SimpleITK as sitk

from radharmony.utils.dicom import PHOTOMETRIC_TAG, is_dicom, stored_value_sum

from .base import RadiologicPreprocessor


class CXRPreprocessor(RadiologicPreprocessor):
    """Preprocessor for chest X-rays and other 2D radiological images.

    Rescales intensity to [-1, 1] using the image's actual min/max, then
    resizes isotropically. It does not crop.

    Supports DICOM, PNG, and JPEG inputs. A file is DICOM when SimpleITK
    reads DICOM tags from it, whatever its file name. A DICOM image gives
    bright bone (MONOCHROME1 is inverted one time).

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
            Preprocessed SimpleITK image. A 1-frame image gives a 2-D image.
        """
        # sitk/GDCM applies the Modality LUT (RescaleSlope/Intercept) automatically.
        # GDCM also inverts MONOCHROME1, so do not invert it again (V-P-1).
        image = self._read(image_path)
        dicom = is_dicom(image_path, image.HasMetaDataKey(PHOTOMETRIC_TAG))  # before the slice drops the tags

        # GDCM reads a 2-D DICOM as a 3-D image with 1 slice. An .mha copy keeps this shape.
        if image.GetDimension() == 3 and image.GetSize()[2] == 1:
            image = image[:, :, 0]
        if dicom:
            image = self._fix_missing_highbit(image, image_path)
            image = self._apply_voi_lut(image, image_path)

        image = self._to_grayscale_float(image)
        image = self._rescale(image, image_path)
        image = self._resize_isotropic(image)

        return image

    @staticmethod
    def _read(image_path: str) -> sitk.Image:
        """Read a file with SimpleITK. Use PIL for a gray plus alpha PNG.

        SimpleITK cannot read a gray plus alpha (LA) PNG. PIL reads it and the
        alpha channel is dropped, as in the ``monai`` reader (D-13, V-P-27).
        """
        try:
            return sitk.ReadImage(image_path)
        except RuntimeError:
            from PIL import Image

            with Image.open(image_path) as im:
                if im.mode not in ("LA", "PA"):
                    raise
                return sitk.GetImageFromArray(np.asarray(im.convert("L")))

    def _fix_missing_highbit(self, image: sitk.Image, image_path: str) -> sitk.Image:
        """Read the pixels with pydicom when the DICOM file has no HighBit tag.

        GDCM then gives a constant image (about 5% of CheXpert-Plus files, see
        ``docs/wiki/explanation/dicom-loading.md``). pydicom takes HighBit = BitsStored - 1.
        The result has the GDCM form (Modality LUT applied, MONOCHROME1 as
        ``s * (M - raw) + b``), as in ``_fix_missing_highbit`` of
        ``radharmony/dataset/base.py``. So the later steps do not change.
        """
        import pydicom

        ds = pydicom.dcmread(image_path, stop_before_pixels=True, force=True)
        if "HighBit" in ds or ds.get("SamplesPerPixel", 1) != 1:
            return image
        try:
            arr = pydicom.dcmread(image_path, force=True).pixel_array.astype("float64")
        except Exception as e:  # for example compressed data with no pydicom decoder
            warnings.warn(f"No HighBit and pydicom cannot decode {image_path}: {e}", stacklevel=2)
            return image
        if ds.get("PhotometricInterpretation", "") == "MONOCHROME1":
            arr = stored_value_sum(ds) - arr
        arr = float(ds.get("RescaleSlope", 1) or 1) * arr + float(ds.get("RescaleIntercept", 0) or 0)
        out = sitk.GetImageFromArray(arr.astype("float32"))
        out.CopyInformation(image)
        return out

    def _apply_voi_lut(self, image: sitk.Image, image_path: str) -> sitk.Image:
        """Apply VOI LUT from DICOM metadata (WindowCenter/Width or LUT sequence).

        Reads only the DICOM header (no pixel data re-read) and uses pydicom's
        apply_voi_lut, which handles both linear windowing and lookup tables.
        GDCM gives a MONOCHROME1 image as ``s * (M - raw) + b``, but the window
        refers to ``s * raw + b``. So this method undoes the GDCM inversion,
        applies the window, and then inverts again. ``M`` comes from
        ``stored_value_sum`` in ``radharmony/utils/dicom.py``. The dataset step
        ``_apply_voi_lut`` in ``radharmony/dataset/base.py`` uses the same rule.
        """
        import pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut

        ds = pydicom.dcmread(image_path, stop_before_pixels=True, force=True)
        if "WindowCenter" not in ds and "VOILUTSequence" not in ds:
            return image
        photometric = ds.get("PhotometricInterpretation", "")
        # The DICOM standard applies a VOI LUT only to MONOCHROME images.
        if photometric not in ("MONOCHROME1", "MONOCHROME2"):
            return image
        mono1 = photometric == "MONOCHROME1"

        arr = sitk.GetArrayFromImage(sitk.Cast(image, sitk.sitkFloat32))
        if mono1:
            s = float(ds.get("RescaleSlope", 1) or 1)
            b = float(ds.get("RescaleIntercept", 0) or 0)
            m = stored_value_sum(ds)
            # For JPEG 2000, GDCM uses 2**BitsAllocated - 1 for unsigned values.
            if m > 0 and (arr.max() - b) / s > m:
                m = 2 ** int(ds.BitsAllocated) - 1
            arr = (s * m + 2 * b) - arr
        arr = apply_voi_lut(arr, ds).astype("float32")
        if mono1:
            arr = arr.max() + arr.min() - arr
        out = sitk.GetImageFromArray(arr)
        out.CopyInformation(image)
        return out

    def _rescale(self, image: sitk.Image, image_path: str = "") -> sitk.Image:
        """Rescale intensity to [-1, 1] using the image's actual min/max.

        A uniform image (min equal to max) becomes all -1, as in the dataset
        transforms (MONAI ``ScaleIntensityRangePercentiles`` gives ``b_min``).
        ``image_path`` names the file in the warning.
        """
        stats = sitk.StatisticsImageFilter()
        stats.Execute(image)
        if stats.GetMinimum() == stats.GetMaximum():
            warnings.warn(f"Uniform image (min == max), the output is all -1: {image_path}", stacklevel=2)
            return sitk.Cast(image, sitk.sitkFloat32) * 0.0 - 1.0
        return sitk.IntensityWindowing(
            image,
            windowMinimum=stats.GetMinimum(),
            windowMaximum=stats.GetMaximum(),
            outputMinimum=-1.0,
            outputMaximum=1.0,
        )
