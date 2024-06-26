"""
This module contains common utilities for image processing.
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


def read_image(relative_dir: str) -> Optional[NDArray]:
    """
    Reads in an image, otherwise returns None

    Args:
        relative_dir (str): Relative path e.g.
        ("data/04_060__150_00_01_00C_230922_3D.tif")

    Returns:
        Optional[NDArray]: Output image
    """

    full_path = Path.cwd().parent / Path(relative_dir)
    img = cv2.imread(str(full_path), cv2.IMREAD_COLOR).astype(np.float32)[
        :, :, [2, 1, 0]
    ]
    img = img / np.max(img)
    return img


def rgb_to_gray(img: NDArray, kernel: Optional[NDArray] = None) -> NDArray:
    """
    Converts image to grayscale through linear combination and rescales to max of 1

    Args:
        img (NDArray): Input image, must be of dtype float32
        kernel (Optional[NDArray], optional): Channel weights, np.float32.
        Defaults to None which is set to [1.0, 1.0, 1.0]

    Returns:
        NDArray: Grayscale image array
    """

    if kernel is None:
        kernel = np.array([1, 1, 1], dtype=np.float32)

    assert img.dtype == np.float32, "Array is not of dtype float32"
    assert kernel.dtype == np.float32, "Kernel is not of dtype float32"
    assert len(img.shape) == 3, "Image must have dimensionality of 3"
    assert len(kernel.shape) == 1, "Kernel must have dimensionality of 1"

    gray = np.dot(img, kernel)
    gray = gray / np.max(gray)
    return gray


def print_image(img: NDArray, axis: bool = False, color_bar: bool = False) -> None:
    """
    Plots/prints the image

    Args:
        img (NDArray): Input image
        axis (bool, optional): Print axis or not. Defaults to False.
        color_bar (bool, optional): Print a color bar or not. Defaults to False.
    """

    # Downsample the image
    w, h = img.shape[0], img.shape[1]

    if h > 2000 or w > 2000:
        new_w = int(w // (h / 2000))
        img = cv2.resize(img, (2000, new_w), interpolation=cv2.INTER_AREA)

    plt.imshow(img, cmap="gray")

    if not axis:
        plt.axis("off")

    if color_bar:
        plt.colorbar()
