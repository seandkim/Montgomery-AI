import math
from typing import List, Optional
import cv2
import os
import torch

import numpy as np
import matplotlib.pyplot as plt

# region verbose
VERBOSE = False


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
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

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
    image: np.array, lines: List[List[int]], title: str = "", gray=False
):
    plt.figure(figsize=(10, 10))
    if gray:
        plt.imshow(image, cmap="gray")
    else:
        plt.imshow(image)

    for line in lines[:20]:
        x1, y1, x2, y2 = line
        plt.plot([x1, x2], [y1, y2], "r-", markersize=1)
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
    # Compute the minimum area bounding rectangle
    rect = cv2.minAreaRect(cnt)  # rect = ((cx, cy), (width, height), angle)
    (width, height) = rect[1]
    # If width or height is zero, shape can't be formed
    if width == 0 or height == 0:
        return 0.0
    rect_area = width * height
    # Compute the rectangularity score
    score = area / rect_area
    return score


def dilate_and_erode(image, skip_dilation=False, iterations=1):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if not skip_dilation:
        image = cv2.dilate(image, kernel, iterations=iterations)
    image = cv2.erode(image, kernel, iterations=iterations)
    return image


def run_hough_line(image_binary):
    lines = cv2.HoughLinesP(
        image_binary,
        rho=1,
        theta=np.pi / 180,
        threshold=1,
        minLineLength=1,  # Adjust based on fret length in image
        maxLineGap=100,  # Adjust tolerance for gaps in a line
    )

    # 'lines' will be an array of shape (N,1,4), where each line is represented as [x1,y1,x2,y2].
    if lines is not None:
        lines = lines[:, 0, :]  # reshape to (N,4)
    else:
        lines = np.empty((0, 4), dtype=np.float32)
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


class Pitch:
    NOTE_NAME_ORDER = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    def __init__(self, note_name: str, octave: int):
        self.note_name = note_name
        self.octave = octave

    def __init__(self, as_str: str):
        if len(as_str) < 2 or len(as_str) > 3:
            raise ValueError(f"Invalid pitch string: {as_str}")

        as_str = as_str.replace("\u266F", "#")  # "♯" symbol
        if as_str[:-1].upper() not in Pitch.NOTE_NAME_ORDER:
            raise ValueError(f"Invalid note name: {as_str[:-1]}")
        self.note_name = as_str[:-1].upper()
        if not as_str[-1].isdigit() or int(as_str[-1]) < 0:
            raise ValueError(f"Invalid octave: {as_str[-1]}")
        self.octave = int(as_str[-1])

    def __repr__(self):
        return f"{self.note_name}{self.octave}"

    def to_int(self) -> int:
        return Pitch.NOTE_NAME_ORDER.index(self.note_name) + 12 * (self.octave)

    def __eq__(self, other: "Pitch") -> bool:
        return self.to_int() == other.to_int()

    def __lt__(self, other: "Pitch") -> bool:
        return self.to_int() < other.to_int()

    def __le__(self, other: "Pitch") -> bool:
        return self.to_int() <= other.to_int()

    def __gt__(self, other: "Pitch") -> bool:
        return self.to_int() > other.to_int()

    def __ge__(self, other: "Pitch") -> bool:
        return self.to_int() >= other.to_int()

    def subtract(self, other: "Pitch") -> int:
        return self.to_int() - other.to_int()


class GuitarTab:
    MAX_FRET_INDEX = 24
    BASE_STRINGS = [
        Pitch("E2"),
        Pitch("A2"),
        Pitch("D3"),
        Pitch("G3"),
        Pitch("B3"),
        Pitch("E4"),
    ]

    def __init__(self, string_index: int, fret_index: int):
        self.string_index = string_index
        self.fret_index = fret_index

    def __repr__(self):
        return f"{GuitarTab.BASE_STRINGS[self.string_index]}: {self.fret_index}"

    def possible_tabs(pitch: Pitch):
        possible = []
        for idx, base in enumerate(GuitarTab.BASE_STRINGS):
            diff = pitch.subtract(base)
            if 0 < diff and diff <= GuitarTab.MAX_FRET_INDEX:
                possible.append(GuitarTab(idx, diff))
        return possible


def tabs2string(tabs: List[GuitarTab]):
    positions_per_string = [[f"{s.note_name} "] for s in GuitarTab.BASE_STRINGS]
    positions_per_string[-1][0] = positions_per_string[-1][0].lower()
    for tab in tabs:
        for string_idx in range(len(GuitarTab.BASE_STRINGS)):
            if string_idx == tab.string_index:
                fret_index = str(tab.fret_index)
                if (len(fret_index)) == 1:
                    fret_index = f"-{fret_index}"
                positions_per_string[string_idx].append(fret_index)
            else:
                positions_per_string[string_idx].append("--")

    return "\n".join(["--".join(positions) for positions in positions_per_string])


def test_tabs2string():
    tabs = [GuitarTab(2, 12), GuitarTab(4, 15), GuitarTab(3, 14), GuitarTab(3, 12)]
    print(tabs2string(tabs))


if __name__ == "__main__":
    test_tabs2string()
    print(GuitarTab.possible_tabs(Pitch("E3")))
