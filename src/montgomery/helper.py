import cv2
import math
import os
import torch
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# region verbose
VERBOSE = False


def print_error(*args, **kwargs):
    RED = "\033[31m"
    RESET = "\033[0m"
    message = " ".join(map(str, args))
    print(f"{RED}{message}{RESET}", **kwargs)


def print_verbose(*args, **kwargs):
    if VERBOSE:
        GREEN = "\033[32m"
        RESET = "\033[0m"
        message = " ".join(map(str, args))
        print(f"{GREEN}{message}{RESET}", **kwargs)


VERBOSE = os.environ.get("PYTHON_VERBOSE_MODE")
if VERBOSE is not None and VERBOSE.lower() == "true":
    VERBOSE = True
    print("VERBOSE mode is enabled")
# endregion


class Point:
    def __init__(self, x: float, y: float, z: float = None):
        self.x = x
        self.y = y
        self.z = z

    def to_coordinates(self) -> np.ndarray:
        coordinates = [self.x, self.y]
        if self.z != None:
            coordinates.append(self.z)
        return np.array(coordinates)

    @staticmethod
    def from_coordinates(coordinates: List[float]):
        if len(coordinates) == 2:
            return Point(coordinates[0], coordinates[1])
        elif len(coordinates) == 3:
            return Point(coordinates[0], coordinates[1], coordinates[2])
        else:
            raise ValueError("Coordinate must be of length 2 or 3")

    def rotate_ccw(self, angle_in_degree, cx, cy):
        d = -math.radians(angle_in_degree)
        X = self.x - cx
        Y = self.y - cy
        Z = self.z
        Xr = X * math.cos(d) - Y * math.sin(d)
        Yr = X * math.sin(d) + Y * math.cos(d)
        x_rot = Xr + cx
        y_rot = Yr + cy
        return Point(x_rot, y_rot, self.z)


def show_image(image: np.array, gray=False):
    plt.figure(figsize=(10, 10))
    if gray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)
    plt.axis("on")
    plt.show(block=True)


def show_image_with_point(
    image: np.array, points: List[Point], title: str = "", gray=False
):
    plt.figure(figsize=(10, 10))
    if gray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    for point in points:
        plt.plot(point.x, point.y, "ro")
    plt.axis("on")
    plt.title(title)
    plt.show(block=True)


def show_image_with_lines(
    image: np.ndarray, lines: List[List[int]], title: str = "", gray=False
):
    plt.figure(figsize=(10, 10))
    if gray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    for line in lines:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], "r-", markersize=1)
    plt.axis("on")
    plt.title(title)
    plt.show(block=True)


def show_image_with_vertical_lines(
    image: np.ndarray, vertical_lines_x: List[int], title: str = ""
):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap="gray")

    for x in vertical_lines_x:
        plt.axvline(x=x, color="r", linestyle="--")

    plt.axis("on")
    plt.title(title)
    plt.show(block=True)


def setup_torch_device() -> torch.device:
    # if using Apple MPS, fall back to CPU for unsupported ops
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    np.random.seed(3)
    return device


def get_bounding_box(image_binary: np.array):
    coords = np.column_stack(np.where(image_binary > 0))
    if coords.size == 0:
        raise ValueError("image_binary is empty.")
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    return x_min, y_min, x_max, y_max


# e.g. 3 o'clock needle => 0, 12 o'clock needle => 90
# Use PCA to find orientation
def get_angle_from_positive_x_axis(image_binary: np.ndarray) -> np.float64:
    coords = np.column_stack(np.where(image_binary > 0))
    if coords.shape[0] == 0:
        raise ValueError("image is empty.")
    mean = np.mean(coords, axis=0)
    centered = coords - mean
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvecs = eigvecs[:, order]
    angle_in_degree = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    return 180 - angle_in_degree


def rotate_ccw(
    image: np.ndarray,
    angle_in_degree: np.float64,
    center: Optional[tuple[int, int]] = None,
) -> np.ndarray:
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle_in_degree, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rotated


# Created by ChatGPT
def rectangularity_score(mask: np.ndarray):
    """
    Compute how close the shape in a binary mask is to a rectangle.
    mask: 2D np.array (0 and 1 values), shape (H, W)
    Returns a float (0 < score <= 1), where 1 = perfect rectangle
    """
    # Convert mask to uint8 image
    mask_u8 = (mask * 255).astype(np.uint8)
    # Find contours
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return 0.0  # No shape found
    # Assume the largest contour represents the shape
    cnt = max(contours, key=cv2.contourArea)
    # Compute area of the shape
    area = cv2.contourArea(cnt)
    if area == 0:
        return 0.0
    rect = cv2.minAreaRect(cnt)
    (width, height) = rect[1]
    if width == 0 or height == 0:
        return 0.0
    rect_area = width * height
    # Compute the rectangularity score
    score = area / rect_area
    return score


def dilate(image, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(image, kernel, iterations=iterations)


def erode(image, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.erode(image, kernel, iterations=iterations)


def dilate_and_erode(image, iterations=1):
    return erode(image, dilate(image, iterations))


def run_hough_line(image_binary):
    lines = cv2.HoughLinesP(
        image_binary,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=1,  # Adjust based on fret length in image
        maxLineGap=100,  # Adjust tolerance for gaps in a line
    )

    # 'lines' will be an array of shape (N,1,4), where each line is represented as [x1,y1,x2,y2].
    if lines is not None:
        lines = lines[:, 0, :]  # reshape to (N,4)
    else:
        lines = np.empty((0, 4), dtype=np.float32)
    print_verbose(f"run_hough_line detected {len(lines)} lines")
    return lines


def is_vertical(line, tolerance=10):
    x1, y1, x2, y2 = line
    angle = math.atan2((y2 - y1), (x2 - x1))
    # Normalize angle between 0 and π
    if angle < 0:
        angle += math.pi

    # Check if angle is near vertical (close to π/2)
    # Allow a small tolerance, say ±5°
    tolerance = tolerance * math.pi / 180
    return abs(angle - math.pi / 2) < tolerance


def find_vertical_sum_peaks(
    image: np.ndarray,
    height: int = 1000,
    distance: int = 25,
    prominence: int = 500,
    show_image: bool = False,
):
    column_sum = np.sum(image, axis=0)

    peaks, properties = find_peaks(
        column_sum, height=height, distance=distance, prominence=prominence
    )

    # Filter out peaks that are outside the core fretboard part
    valid_peaks = []
    neighbors_sum_threshold = 10000
    neighbor_window = 5
    for peak in peaks:
        left_bound = max(0, peak - neighbor_window)
        right_bound = min(len(column_sum) - 1, peak + neighbor_window)
        if peak - neighbor_window < 0:
            right_bound = peak + (neighbor_window * 2 - peak)
        elif peak + neighbor_window > len(column_sum) - 1:
            left_bound = peak - (neighbor_window * 2 - (len(column_sum) - peak - 1))
        neighbors_sum = sum(column_sum[left_bound:right_bound])
        # print_verbose(f"Peak at {peak}: neighboring sum is {neighbors_sum}")
        if neighbors_sum > neighbors_sum_threshold:
            valid_peaks.append(peak)
        else:
            print_verbose(
                f"Remove peak at {peak} since the neighboring sum is {neighbors_sum} < {neighbors_sum_threshold}"
            )
    peaks = np.array(valid_peaks)

    peaks = np.sort(peaks)
    print_verbose("Found peaks at columns:", peaks)

    if show_image:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(column_sum, label="Column sum (edge intensity)")
        ax1.plot(peaks, column_sum[peaks], "x", label="Detected peaks")
        ax1.set_title("Vertical Projection of Edges and Detected Peaks")
        ax1.set_xlabel("Column Index (x-coordinate)")
        ax1.set_ylabel("Edge Sum")
        ax1.legend()

        ax2.imshow(image, cmap="gray")
        for x in peaks:
            ax2.axvline(x=x, color="r", linestyle="--")
        ax2.set_title("Image with Detected Peaks")
        ax2.axis("on")

        plt.tight_layout()
        plt.show(block=True)

        # plt.figure(figsize=(10, 4))
        # plt.plot(column_sum, label="Column sum (edge intensity)")
        # plt.plot(peaks, column_sum[peaks], "x", label="Detected peaks")
        # plt.title("Vertical Projection of Edges and Detected Peaks")
        # plt.xlabel("Column Index (x-coordinate)")
        # plt.ylabel("Edge Sum")
        # plt.legend()
        # plt.show(block=True)

        # plt.figure(figsize=(10, 10))
        # plt.imshow(image, cmap="gray")

        # for x in peaks:
        #     plt.axvline(x=x, color="r", linestyle="--")

        # plt.axis("on")
        # plt.show(block=True)
    return peaks
