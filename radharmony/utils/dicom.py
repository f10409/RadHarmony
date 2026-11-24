"""DICOM helpers shared by the dataset loader and the CXR preprocessor."""

# DICOM tag (0028,0004) Photometric Interpretation. The GDCM reader of SimpleITK puts
# it into the image information of every DICOM file, with any file name (D-13).
PHOTOMETRIC_TAG = "0028|0004"

# SimpleITK keeps the DICOM tags when it saves an image in these formats. These files
# are never DICOM files.
_COPY_FORMATS = (".mha", ".mhd", ".nrrd", ".nhdr", ".nii", ".nii.gz")


def is_dicom(path, has_photometric_tag: bool) -> bool:
    """Return True for a DICOM file (D-13).

    Args:
        path: the image file.
        has_photometric_tag: True when the image information has ``PHOTOMETRIC_TAG``.

    Returns:
        True when the tag is there and the file is not an ``.mha``, ``.nrrd`` or NIfTI
        copy of a DICOM file.
    """
    return has_photometric_tag and not str(path).lower().endswith(_COPY_FORMATS)


def stored_value_sum(ds) -> int:
    """Return M, the largest plus the smallest stored value of a DICOM file.

    GDCM gives a MONOCHROME1 image as ``s * (M - raw) + b``. For unsigned
    values, M is ``2**BitsStored - 1``. For signed values, M is
    ``(2**(n-1) - 1) - 2**(n-1) = -1``.
    """
    if ds.get("PixelRepresentation", 0) == 1:
        return -1
    return 2 ** int(ds.BitsStored) - 1
