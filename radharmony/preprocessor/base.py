"""Base medical image preprocessor with shared utilities."""

import os

import pandas as pd
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class RadiologicPreprocessor:
    """Base class for medical image preprocessing.

    Provides shared utilities for intensity normalization, foreground cropping,
    and isotropic resizing. Subclass for modality-specific pipelines.

    Args:
        target_long_side: Desired pixel/voxel count for the longest axial dimension.
        interpolation: SimpleITK interpolation method (default: sitkLinear).
    """

    def __init__(
        self,
        target_long_side: int = 112,
        interpolation=sitk.sitkLinear,
    ):
        self.target_long_side = target_long_side
        self.interpolation = interpolation

    def __call__(self, image_path: str) -> sitk.Image:
        raise NotImplementedError("Subclasses must implement __call__.")

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _fix_monochrome(self, image: sitk.Image) -> sitk.Image:
        """Invert pixel values for DICOM MONOCHROME1 images."""
        try:
            if image.GetMetaData("0028|0004").strip() == "MONOCHROME1":
                return sitk.InvertIntensity(image)
        except RuntimeError:
            pass
        return image

    def _to_grayscale_float(self, image: sitk.Image) -> sitk.Image:
        """Convert to single-channel float32, applying ITU-R 601 luma for RGB/RGBA inputs."""
        if image.GetNumberOfComponentsPerPixel() > 1:
            r = sitk.VectorIndexSelectionCast(image, 0, sitk.sitkFloat32)
            g = sitk.VectorIndexSelectionCast(image, 1, sitk.sitkFloat32)
            b = sitk.VectorIndexSelectionCast(image, 2, sitk.sitkFloat32)
            return sitk.Cast(r * 0.2989 + g * 0.5870 + b * 0.1140, sitk.sitkFloat32)
        return sitk.Cast(image, sitk.sitkFloat32)

    def _crop_foreground(self, image: sitk.Image) -> sitk.Image:
        """Crop image to the bounding box of the largest connected foreground component.

        Uses Otsu thresholding followed by morphological opening to find the
        dominant anatomy region.  Returns the original image if no foreground
        is detected.

        Args:
            image: Windowed 2D SimpleITK image.

        Returns:
            Cropped SimpleITK image, or the input unchanged if no labels found.
        """
        mask = sitk.OtsuThreshold(image, 0, 1)
        mask = sitk.BinaryMorphologicalOpening(mask, [3, 3])

        labels = sitk.ConnectedComponent(mask)
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(labels)

        if len(stats.GetLabels()) == 0:
            return image

        largest_label = max(stats.GetLabels(), key=lambda l: stats.GetNumberOfPixels(l))
        bbox = stats.GetBoundingBox(largest_label)  # (x, y, w, h) for 2D

        return sitk.RegionOfInterest(image, [bbox[2], bbox[3]], [bbox[0], bbox[1]])

    def _crop_2d(self, image: sitk.Image) -> sitk.Image:
        """Crop a 2D image to the largest foreground region."""
        return self._crop_foreground(image)

    def _crop_3d(self, image: sitk.Image) -> sitk.Image:
        """Crop a 3D volume axially using the center slice for foreground detection.

        Applies foreground cropping on the center axial slice and extends the
        resulting 2D bounding box across the full Z depth.

        Args:
            image: Windowed 3D SimpleITK image.

        Returns:
            Cropped SimpleITK image, or the input unchanged if no labels found.
        """
        size = image.GetSize()
        center_slice = image[:, :, size[2] // 2]

        cropped_slice = self._crop_foreground(center_slice)

        # Recover the 2D bounding box from the crop origin/size delta
        origin_2d = cropped_slice.GetOrigin()
        full_origin = image.GetOrigin()
        spacing = image.GetSpacing()

        start_x = max(0, int(round((origin_2d[0] - full_origin[0]) / spacing[0])))
        start_y = max(0, int(round((origin_2d[1] - full_origin[1]) / spacing[1])))
        crop_x, crop_y = cropped_slice.GetSize()
        crop_x = min(crop_x, size[0] - start_x)
        crop_y = min(crop_y, size[1] - start_y)

        return sitk.RegionOfInterest(
            image,
            [crop_x, crop_y, size[2]],
            [start_x, start_y, 0],
        )

    def _resize_isotropic(self, image: sitk.Image) -> sitk.Image:
        """Resample to isotropic spacing so the longest side equals target_long_side.

        Works for both 2D and 3D images.

        Args:
            image: SimpleITK image to resample.

        Returns:
            Resampled SimpleITK image with uniform spacing across all axes.
        """
        input_size = image.GetSize()
        input_spacing = image.GetSpacing()

        idx_long = 0 if input_size[0] >= input_size[1] else 1
        new_spacing = (
            input_size[idx_long] * input_spacing[idx_long]
        ) / self.target_long_side

        ndim = image.GetDimension()
        out_size = [
            int(round((input_size[i] * input_spacing[i]) / new_spacing))
            for i in range(ndim)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(out_size)
        resampler.SetOutputSpacing([new_spacing] * ndim)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(self.interpolation)
        resampler.SetDefaultPixelValue(-1.0)

        return resampler.Execute(image)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def run_batch(
        self,
        df: "pd.DataFrame",
        base_output_dir: str,
        base_image_dir: str = None,
        num_workers: int = 32,
    ) -> None:
        """Preprocess all images in a DataFrame and write results to disk.

        The DataFrame must contain an ``image_path`` column with either absolute
        or relative paths.  Pass ``base_image_dir`` when paths are relative.

        Args:
            df: DataFrame with image paths.
            base_output_dir: Directory where preprocessed files are written.
            base_image_dir: Root directory prepended to relative ``image_path``
                values.  Ignored when paths are already absolute.
            num_workers: Number of parallel worker processes (default: 32).
        """
        os.makedirs(base_output_dir, exist_ok=True)

        worker_args = [(self, row, base_output_dir, base_image_dir) for row in df.to_dict("records")]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_row, a) for a in worker_args]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                pass


# Module-level for multiprocessing pickling
def _process_row(args):
    """Preprocess one image and save to disk.

    Args:
        args: Tuple of (RadiologicPreprocessor, row dict, output_dir, base_image_dir).

    Returns:
        True on success, or an error message string on failure.
    """
    preprocessor, row, output_dir, base_image_dir = args
    image_path = row["image_path"]
    if base_image_dir is not None:
        image_path = os.path.join(base_image_dir, image_path)
    try:
        vol = preprocessor(image_path)
        basename = os.path.basename(image_path)
        stem = (
            basename[:-7]
            if basename.endswith(".nii.gz")
            else os.path.splitext(basename)[0]
        )
        output_filename = stem + ".nii"
        sitk.WriteImage(vol, os.path.join(output_dir, output_filename))
        return True
    except Exception as e:
        return f"Error with {image_path}: {e}"
