"""Small hand-made images for tests (no NAS data).

A DICOM file made here follows the DICOM standard (PS3.3 C.7.6.3). So a test
gets the expected pixels from pydicom, not from the code under test.
"""

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid


def write_dicom(
    path,
    stored: np.ndarray,
    photometric: str = "MONOCHROME2",
    bits_stored: int = 12,
    window: tuple[float, float] | None = None,
    rescale: tuple[float, float] | None = None,
    pixel_spacing: tuple[float, float] = (0.2, 0.2),
    signed: bool = False,
) -> str:
    """Write a 1-frame DICOM file.

    Args:
        path: Output file. Any file name works (``.dcm``, ``.dicom`` or none).
        stored: (rows, cols) stored pixel values. A (rows, cols, 3) uint8 array
            gives an 8-bit RGB file, and ``photometric`` and ``bits_stored`` are
            not used.
        photometric: ``"MONOCHROME1"`` or ``"MONOCHROME2"``.
        bits_stored: BitsStored tag. BitsAllocated is 16.
        window: (WindowCenter, WindowWidth), or None for no VOI LUT.
        rescale: (RescaleSlope, RescaleIntercept), or None for no Modality LUT.
        pixel_spacing: PixelSpacing tag (row, column) in mm.
        signed: PixelRepresentation 1 (signed values) instead of 0.

    Returns:
        The path as a string.
    """
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CR"
    ds.PatientID = "p1"
    ds.Rows, ds.Columns = stored.shape[:2]
    if stored.ndim == 3:
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
        ds.PhotometricInterpretation = "RGB"
        ds.BitsAllocated = ds.BitsStored = 8
    else:
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = photometric
        ds.BitsAllocated = 16
        ds.BitsStored = bits_stored
    ds.HighBit = ds.BitsStored - 1
    ds.PixelRepresentation = int(signed)
    ds.PixelSpacing = list(pixel_spacing)
    if window is not None:
        ds.WindowCenter, ds.WindowWidth = window
    if rescale is not None:
        ds.RescaleSlope, ds.RescaleIntercept = rescale
    if stored.ndim == 3:
        ds.PixelData = stored.astype(np.uint8).tobytes()
    else:
        ds.PixelData = stored.astype(np.int16 if signed else np.uint16).tobytes()
    ds.save_as(str(path), enforce_file_format=True)
    return str(path)


def dicom_display_values(path) -> np.ndarray:
    """Return the pixels of a DICOM file as a viewer shows them, bone bright.

    The order is the DICOM pipeline: Modality LUT, then VOI LUT, then
    MONOCHROME1 inversion (the lowest value is white).
    """
    ds = pydicom.dcmread(str(path))
    values = apply_modality_lut(ds.pixel_array, ds)
    values = apply_voi_lut(values, ds).astype(np.float64)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        values = -values
    return values


def to_unit_range(values: np.ndarray) -> np.ndarray:
    """Scale an array to [-1, 1] with its min and max."""
    values = values.astype(np.float64)
    return 2 * (values - values.min()) / (values.max() - values.min()) - 1
