"""Image sizes for the VinDr-CXR boxes, from the DICOM headers (D-07)."""
from __future__ import annotations

import warnings

from .._dicom_tags import read_dicom_tags


def image_sizes(base_image_dir: str | None, image_ids, name: str) -> dict[str, tuple[int, int]]:
    """Return ``{image_id: (Rows, Columns)}`` for the images that have a box.

    A box needs the image size, so an image with no readable header gets no box.
    With no ``base_image_dir``, every image loses its boxes, and a warning says so
    (V-H3-35). ``read_dicom_tags`` warns about a file that cannot be read.

    Args:
        base_image_dir: The folder of the ``<image_id>.dicom`` files.
        image_ids: The ids of the images with a box.
        name: Dataset name for the warning.
    """
    image_ids = list(image_ids)
    if not base_image_dir:
        if image_ids:
            warnings.warn(
                f"{name}: bbox_csv_path is set, but base_image_dir is not. A box needs the image size from "
                f"the DICOM header, so {len(image_ids)} images get no box (D-07).",
                UserWarning,
                stacklevel=3,
            )
        return {}
    headers = read_dicom_tags(base_image_dir, [f"{i}.dicom" for i in image_ids], ["Rows", "Columns"])
    return {path[: -len(".dicom")]: (int(tags["Rows"]), int(tags["Columns"]))
            for path, tags in headers.items() if tags["Rows"] is not None and tags["Columns"] is not None}
