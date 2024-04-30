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


def remove_background(img: NDArray) -> NDArray:
    """
    Returns np array mask of the specimen.  With specimen 1 and background 0 in uint16 format.
    """
    channel_letter = args.background_channel
    binary_color = args.binary_color
    path = img_path
    threshold = args.threshold

    img_color = img
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

    eroded_image_np = sitk.GetArrayFromImage(eroded_image)
    eroded_image_np = np.expand_dims(eroded_image_np, axis=-1)

    return eroded_image_np


def find_fibers(img: NDArray) -> NDArray:
    print("Converting image to HSV")
    hue = skimage.color.rgb2hsv(img)[:, :, 0]

    print("Running Otsu on hue channel")
    thresh = skimage.filters.threshold_otsu(hue)
    hue_otsu = (hue > thresh).astype(np.uint8)

    return np.expand_dims(hue_otsu, axis=-1)


def save_heatmap(
    fiber_mask: NDArray, background_mask: NDArray, path: Path, sigma: int = 20
) -> None:
    """
    Create a heatmap from a grayscale image and save it to disk.
    """

    # Apply blur
    fiber_mask_float = fiber_mask.astype(np.float32)
    heat_map = skimage.filters.gaussian(fiber_mask_float, sigma=20)
    heat_map_u8 = (heat_map * 255).astype(np.uint8)

    # Change to color space
    heat_map_colored = cv2.applyColorMap(heat_map_u8, cv2.COLORMAP_HOT)

    # Apply background mask
    heat_map_colored = heat_map_colored * background_mask

    # Save temperature map to disk
    save_img(path, heat_map_colored)


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

    # Read the image from disk
    img = utils.read_image(img_path)

    # Remove background
    print("Finding background mask")
    background_mask = remove_background(img)

    # Save background mask to disk
    background_mask_path = results_path / "background_mask.tif"
    print(f"\nSaving background mask to disk: {background_mask_path}")
    save_img(background_mask_path, background_mask)

    # Finding fibers
    print("\nFinding fibers in image")
    fiber_mask = find_fibers(img)
    fiber_mask = fiber_mask * background_mask

    # Save fiber mask to disk
    fiber_mask_pth = results_path / "fiber_mask.tif"
    print(f"\n\nSaving image without background to disk: {fiber_mask_pth}")
    save_img(fiber_mask_pth, fiber_mask)

    # Create heatmap
    print("\nCreating heatmap 1")
    save_heatmap(fiber_mask, background_mask, results_path / "heatmap_5.tif", sigma=5)

    # Create heatmap
    print("\nCreating heatmap 2")
    save_heatmap(fiber_mask, background_mask, results_path / "heatmap_10.tif", sigma=10)

    # Create heatmap
    print("\nCreating heatmap 3")
    save_heatmap(fiber_mask, background_mask, results_path / "heatmap_20.tif", sigma=20)
