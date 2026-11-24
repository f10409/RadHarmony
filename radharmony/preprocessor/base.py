"""Base medical image preprocessor with shared utilities."""

import os
import re

import numpy as np
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

        # Find the start index of the crop. The index conversion uses the direction
        # matrix, so a flipped axis gives the right box (V-P-5).
        start_x, start_y = center_slice.TransformPhysicalPointToIndex(cropped_slice.GetOrigin())
        start_x, start_y = max(0, start_x), max(0, start_y)
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

        # The long side by its length in mm, not by its pixel count (V-P-11).
        idx_long = 0 if input_size[0] * input_spacing[0] >= input_size[1] * input_spacing[1] else 1
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
        # Keep the image centered: the output covers the same area as the input (D-43, Q-20).
        direction = np.array(image.GetDirection()).reshape(ndim, ndim)
        shift = np.array([(new_spacing - input_spacing[i]) / 2 for i in range(ndim)])
        resampler.SetOutputOrigin((np.array(image.GetOrigin()) + direction @ shift).tolist())
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

        Each output file is :func:`output_path` of its ``image_path``.

        Raises:
            ValueError: before any work, if an ``image_path`` has a ``..`` part (V-P-22).
            RuntimeError: after the batch, if an image failed (V-P-3).
        """
        for p in df["image_path"]:
            output_path(p, base_output_dir)  # raises on a '..' part, before any file is written
        os.makedirs(base_output_dir, exist_ok=True)

        worker_args = [(self, row, base_output_dir, base_image_dir) for row in df.to_dict("records")]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_process_row, a) for a in worker_args]
            results = [f.result() for f in tqdm(as_completed(futures), total=len(futures), desc="Processing")]

        # A failed image must not leave image_path on a copy that does not exist (V-P-3).
        errors = [r for r in results if r is not True]
        if errors:
            raise RuntimeError(
                f"{len(errors)} of {len(results)} images failed. The other copies are in "
                f"{base_output_dir}. First error: {errors[0]}"
            )


def output_path(image_path: str, output_dir: str) -> str:
    """Return the output file of one image: its path below ``output_dir``, plus the ending ``.nii``.

    The folders and the file ending of ``image_path`` stay. So 2 images with the same file name
    get 2 files (V-P-4), and ``a/1.png`` and ``a/1.jpg`` give ``a/1.png.nii`` and ``a/1.jpg.nii`` (V-P-22).

    Raises:
        ValueError: if ``image_path`` has a ``..`` part. The file could then be outside ``output_dir`` (V-P-22).
    """
    rel = str(image_path).lstrip("/\\")
    if ".." in re.split(r"[/\\]", rel):
        raise ValueError(f"image_path {image_path!r} has a '..' part, so its copy can be outside {output_dir}.")
    return os.path.join(output_dir, rel + ".nii")


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
        out = output_path(row["image_path"], output_dir)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        sitk.WriteImage(vol, out)
        return True
    except Exception as e:
        return f"Error with {image_path}: {e}"
