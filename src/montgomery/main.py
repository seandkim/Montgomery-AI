import cv2
import numpy as np
import os
from PIL import Image
from typing import List, Optional
import matplotlib.pyplot as plt

from . import helper
from . import sam2_helper
from . import mediapipe_helper as mp_helper

from .helper import print_verbose
from .sam2_helper import SAM2MaskResult
from .mediapipe_helper import HandResult, Handedness


def run_canny_edge(image_rgb: np.ndarray, blur=False, show_image=False) -> np.ndarray:
    result = image_rgb.copy()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    if blur:
        result = cv2.GaussianBlur(result, (5, 5), 1.4)
    result = cv2.Canny(result, 100, 200)

    if show_image:
        plt.subplot(121), plt.imshow(image_rgb, cmap="gray")
        plt.title("Original Image"), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(result, cmap="gray")
        plt.title("Canny Edges"), plt.xticks([]), plt.yticks([])
        plt.show(block=True)

    return result


# region fretboard


def select_fretboard_mask_result(mask_results: List[SAM2MaskResult]) -> SAM2MaskResult:
    # TODO: implement
    return mask_results[1]


def get_fretboard_mask_result(
    image_rgb: np.ndarray, show_image=False
) -> SAM2MaskResult:
    device = helper.setup_torch_device()
    mask_results = sam2_helper.run_sam2(device, image_rgb, input_point, input_label)
    if show_image:
        sam2_helper.show_mask(
            image_rgb,
            mask_results,
            point_coords=input_point,
            input_labels=input_label,
            borders=True,
            block=True,
        )
    return select_fretboard_mask_result(mask_results)


# endregion


def get_hand_result(image_rgb: np.ndarray, save_image=False) -> mp_helper.HandResult:
    min_confidence = 0.1
    left_hand_result = None
    with mp_helper.initialize_mp_hands(min_confidence=min_confidence) as hands:
        mp_hand_results = mp_helper.run_mp_hands(hands, image_rgb)
        if mp_hand_results == None:
            print_verbose(
                f"hands not detected: file={file}, min_confidence={min_confidence}"
            )
            exit

        for _, mp_hand_result in enumerate(mp_hand_results):
            hand_result = HandResult.from_mediapipe_result(
                mp_hand_result.handedness,
                mp_hand_result.landmarks_normalized,
                image_height=image_rgb.shape[1],
                image_width=image_rgb.shape[0],
            )

            if left_hand_result == None and hand_result.handedness == Handedness.LEFT:
                left_hand_result = hand_result
    return left_hand_result


if __name__ == "__main__":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    file = "./images/raw/guitar.png"
    image_bgr = Image.open(file)
    image_rgb = np.array(image_bgr.convert("RGB"))
    input_point = np.array([[1600, 200]])
    input_label = np.array([1])

    # canny_result = run_canny_edge(image_rgb)

    freboard_mask_result = get_fretboard_mask_result(image_rgb)
    hand_result = get_hand_result(image_rgb)
    helper.show_image_with_point(image_rgb, hand_result.landmarks_normalized)

    # tranform
    orientation = freboard_mask_result.get_orientation()
    freboard_mask_result = freboard_mask_result.rotate(orientation)
    hand_result = hand_result.rotate(orientation)
    # helper.show_image_with_point(image_rgb, hand_result.landmarks_normalized)

    print(hand_result)
