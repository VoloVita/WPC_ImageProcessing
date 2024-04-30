"""
This script analyzes a microscopy image.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import skimage
from numpy.typing import NDArray

import utils


def check_path(path: str) -> Path:
    """
    Checks if the image file exists and creates a directory for the results.

    Args:
        path (str): Image path

    Raises:
        FileNotFoundError: If the image file does not exist.

    Returns:
        Path: Directory for the results.
    """
    img_pth = Path(path)

    if not img_pth.exists():
        raise FileNotFoundError(f"File not found: {path}")

    i = 1
    base_path = Path(img_pth).resolve().parent / Path("results")
    results_path = base_path / (Path(img_pth).stem + f"_{i:03d}")

    while results_path.exists():
        i += 1
        results_path = base_path / (Path(img_pth).stem + f"_{i:03d}")

    results_path.mkdir(parents=True, exist_ok=False)

    return results_path


def new_remove_background():
    channel_letter = args.background_channel
    binary_color = args.binary_color
    path = img_path
    threshold = args.threshold

    img_color = utils.read_image(path)
    img_gray = utils.rgb_to_gray(img_color)

    rgb_channels = {"R": 0, "G": 1, "B": 2}
    hsv_channels = {"H": 0, "S": 1, "V": 2}
    if channel_letter in rgb_channels:
        channel_num = rgb_channels[channel_letter]
        channel_image = img_color[:, :, channel_num]
    elif channel_letter in hsv_channels:
        channel_num = hsv_channels[channel_letter]
        img_hsv = skimage.color.rgb2hsv(img_color)
        channel_image = img_hsv[:, :, channel_num]
    else:
        print("Invalid channel letter provided")

    if binary_color == "B":
        channel_binary = (channel_image < threshold).astype(np.uint16)
    elif binary_color == "W":
        channel_binary = (channel_image > threshold).astype(np.uint16)
    channel_binary = sitk.GetImageFromArray(channel_binary)

    # Masks binary image to disregard black text from label
    img_gray = sitk.GetImageFromArray(img_gray)
    img_gray = sitk.RescaleIntensity(img_gray, 0, 255)
    binary_mask = sitk.BinaryThreshold(
        img_gray, lowerThreshold=0, upperThreshold=25, insideValue=0, outsideValue=1
    )
    mask_filter = sitk.MaskImageFilter()
    channel_binary = mask_filter.Execute(channel_binary, binary_mask)

    # Closes image
    close_filter = sitk.BinaryMorphologicalClosingImageFilter()
    close_filter.SetKernelRadius([45, 45])
    close_filter.SetForegroundValue(1)

    closed_image = close_filter.Execute(channel_binary)

    # Erodes image, given that we can afford to lose edge pixels in order to remove background
    erode_filter = sitk.BinaryErodeImageFilter()
    erode_filter.SetKernelRadius([60, 60])
    erode_filter.SetForegroundValue(1)

    eroded_image = erode_filter.Execute(closed_image)

    # Applies a binary mask to the color image, removing background
    color_image = sitk.GetImageFromArray(img_color, isVector=True)
    mask_image = eroded_image

    binary_mask = sitk.BinaryThreshold(
        mask_image, lowerThreshold=1, upperThreshold=255, insideValue=1, outsideValue=0
    )

    mask_filter = sitk.MaskImageFilter()
    mask_filter.SetMaskingValue(0)
    masked_color_image = mask_filter.Execute(color_image, binary_mask)

    return sitk.GetArrayFromImage(masked_color_image)


def find_fibers(img: NDArray) -> NDArray:
    print("Converting image to HSV")
    hue = skimage.color.rgb2hsv(img)[:, :, 0]

    print("Running Otsu on hue channel")
    thresh = skimage.filters.threshold_otsu(hue)
    hue_otsu = (hue < thresh).astype(np.uint8)

    return np.expand_dims(hue_otsu, axis=-1)


def save_img(path: Path, img: NDArray) -> None:
    """
    Save image to disk.

    Args:
        path (Path): Path to save the image.
        img (NDArray): Image to save.
    """
    img_new = (img * 255).astype(np.uint8)
    # todo: remove this check
    # if len(img_new.shape) == 3:
    #     img_new = img_new[:, :, ::-1]
    img_new = img_new[:, :, ::-1]
    cv2.imwrite(str(path), img_new)


if __name__ == "__main__":
    # Get command line arguments
    parser = argparse.ArgumentParser(description="Analyze a microscopy image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument(
        "-c",
        "--background_channel",
        choices=["R", "G", "B", "H", "S", "V"],
        default="G",
        help="Color for background removal. Options: R, G, B, H, S, V. Default: H",
    )
    parser.add_argument(
        "-b",
        "--binary_color",
        choices=["B", "W"],
        default="B",
        help="Binary color background removal. Default: Black (B)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.55,
        help="Threshold for background removal. Default: 0.55",
    )

    args = parser.parse_args()
    img_path = Path(args.image).resolve()

    results_path = check_path(img_path)

    # Remove background
    print("Removing background from image")
    img_color_no_background = new_remove_background()

    # Save without background to disk
    img_no_bg_path = results_path / "img_color_no_background.tif"
    print(f"\nSaving image without background to disk: {img_no_bg_path}")
    save_img(img_no_bg_path, img_color_no_background)

    # Finding fibers
    print("\nFinding fibers in image")
    fiber_img = find_fibers(img_color_no_background)

    # Save fiber mask to disk
    fiber_mask_pth = results_path / "fiber_mask.tif"
    print(f"\n\nSaving image without background to disk: {img_no_bg_path}")
    save_img(fiber_mask_pth, fiber_img)

    breakpoint()
