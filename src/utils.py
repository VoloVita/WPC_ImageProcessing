from pathlib import Path
from typing import Optional

import cv2
from numpy.typing import NDArray


def read_image(relative_dir: str) -> Optional[NDArray]:
    """
    Reads in an image, otherwise returns None

    Args:
        relative_dir (str): Relative path e.g. ("data/04_060__150_00_01_00C_230922_3D.tif"v)

    Returns:
        Optional[NDArray]: Output image
    """

    full_path = Path.cwd().parent / Path(relative_dir)
    return cv2.imread(str(full_path))
