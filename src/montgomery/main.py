import json
import cv2
import numpy as np
import os
from PIL import Image
from typing import List, Optional
import matplotlib.pyplot as plt

from moviepy import VideoFileClip  # set IMAGEIO_FFMPEG_EXE env var

from . import guitar
from . import helper
from . import sam2_helper
from . import mediapipe_helper as mp_helper
from . import crepe_helper

from .guitar import GuitarTab, Guitar
from .helper import Point, print_error, print_verbose
from .sam2_helper import SAM2MaskResult
from .mediapipe_helper import HandResult, Handedness


def run_canny_edge(
    image_rgb: np.ndarray, skip_blur=False, show_image=False
) -> np.ndarray:
    result = image_rgb.copy()
    result = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    if not skip_blur:
        sigma = 1.4  # Example sigma value
        kernel_size = int(6 * sigma + 1)  # Common choice to cover Gaussian distribution
        result = cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigmaX=sigma)
        # result = cv2.GaussianBlur(result, (5, 5), 1.4)
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
    best_idx, best_score = None, -1
    for idx in range(len(mask_results)):
        mask_result = mask_results[idx]
        score = helper.rectangularity_score(mask_result.mask)
        if score > best_score:
            best_idx, best_score = idx, score
    print_verbose(f"Selected fretboard mask: idx={idx}")
    return mask_results[best_idx]


def get_fretboard_mask_result(
    image_rgb: np.ndarray,
    input_point: np.ndarray,
    show_all_masks=False,
    ignore_not_found=False,
) -> SAM2MaskResult:
    device = helper.setup_torch_device()
    input_label = np.array([1])
    mask_results = sam2_helper.run_sam2(device, image_rgb, input_point, input_label)
    if not ignore_not_found and (mask_results is None or len(mask_results) == 0):
        raise RuntimeError("Mask results not found")
    if show_all_masks:
        for mask_result in mask_results:
            sam2_helper.show_mask(
                image_rgb,
                mask_result,
                point_coords=input_point,
                input_labels=input_label,
                borders=True,
                block=True,
            )
    return select_fretboard_mask_result(mask_results)


# endregion


def get_hand_result(
    image_rgb: np.ndarray, ignore_not_found=False
) -> mp_helper.HandResult:
    min_confidence = 0.1
    with mp_helper.initialize_mp_hands(min_confidence=min_confidence) as hands:
        hand_results = mp_helper.run_mp_hands(hands, image_rgb)
        if not ignore_not_found and (hand_results is None or len(hand_results) == 0):
            print_error(f"Hand could not be detected")
            raise RuntimeError("Hand could not be detected")

        for hand_result in hand_results:
            if hand_result.handedness == Handedness.LEFT:
                return hand_result
    return None


# Visual Montgomery Result (detecting fretboard mask and hand in one image)
class VisMontResult:
    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        canny: np.ndarray,
        peaks_vertical: np.ndarray,
        hand: HandResult,
    ):
        self.image = image
        self.mask = mask
        self.canny = canny
        self.peaks_vertical = peaks_vertical
        self.hand = hand

    def plot_canny_and_fingertips(self, exclude_thumb=False, title=""):
        indices = [0, 1, 2, 3, 4]
        if exclude_thumb:
            indices = [1, 2, 3, 4]
        helper.show_image_with_point(self.canny, self.hand.tips(indices), title=title)


class MontInputs:
    def __init__(
        self,
        sam_input_point: np.ndarray,  # e.g. [100, 200]
        crepe_model: str,
        crepe_duration: float,
        crepe_offset: float,
        crepe_shift_by_half_note: int,
        vertical_sum_height: int,
        vertical_sum_distance: int,
        vertical_sum_prominence: int,
    ):
        if any(
            arg is None
            for arg in [
                sam_input_point,
                crepe_model,
                crepe_duration,
                crepe_offset,
                crepe_shift_by_half_note,
                vertical_sum_height,
                vertical_sum_distance,
                vertical_sum_prominence,
            ]
        ):
            raise ValueError("MontInputs got null argument")

        self.sam_input_point = sam_input_point
        self.crepe_model = crepe_model
        self.crepe_duration = crepe_duration
        self.crepe_offset = crepe_offset
        self.crepe_shift_by_half_note = crepe_shift_by_half_note
        self.vertical_sum_height = vertical_sum_height
        self.vertical_sum_distance = vertical_sum_distance
        self.vertical_sum_prominence = vertical_sum_prominence

    @staticmethod
    def load_from_json_file(json_file):
        with open(json_file, "r") as f:
            data = json.load(f)

        sam = data.get("sam", {})
        sam_input_point = sam.get("input_point", None)
        crepe = data.get("crepe", {})
        crepe_model = crepe.get("model", None)
        crepe_duration = crepe.get("duration", None)
        crepe_offset = crepe.get("offset", None)
        crepe_shift_by_half_note = crepe.get("shift_by_half_note", None)
        vertical_sum = data.get("vertical_sum", {})
        vertical_sum_height = vertical_sum.get("height", None)
        vertical_sum_distance = vertical_sum.get("distance", None)
        vertical_sum_prominence = vertical_sum.get("prominence", None)
        return MontInputs(
            sam_input_point,
            crepe_model,
            crepe_duration,
            crepe_offset,
            crepe_shift_by_half_note,
            vertical_sum_height,
            vertical_sum_distance,
            vertical_sum_prominence,
        )


def run_vismont(
    image_rgb,
    fretboard_mask_result: SAM2MaskResult,
    mont_inputs: MontInputs,
    show_image: bool = False,
) -> Optional[VisMontResult]:
    hand_result: HandResult = get_hand_result(image_rgb)
    if hand_result == None:
        return None

    angle_to_rotate_ccw = fretboard_mask_result.get_angle_from_positive_x_axis() - 90
    image_rotated = helper.rotate_ccw(
        image_rgb,
        angle_to_rotate_ccw,
        (image_rgb.shape[1] // 2, image_rgb.shape[0] // 2),
    )
    mask_rotated = fretboard_mask_result.rotate_ccw(angle_to_rotate_ccw)
    hand_rotated = hand_result.rotate_ccw(angle_to_rotate_ccw)
    image_rotated_masked = mask_rotated.apply_to_image(image_rotated)
    canny = run_canny_edge(image_rotated_masked, skip_blur=True)
    peaks_vertical = helper.find_vertical_sum_peaks(
        canny,
        height=mont_inputs.vertical_sum_height,
        distance=mont_inputs.vertical_sum_distance,
        prominence=mont_inputs.vertical_sum_prominence,
        show_image=show_image,
    )
    return VisMontResult(
        image_rgb, mask_rotated.mask, canny, peaks_vertical, hand_rotated
    )


def run_fullmont(
    video_file: str, audio_file: str, mont_inputs: MontInputs, show_image: bool = False
):
    video = VideoFileClip(video_file)
    audio_pitch_infos = crepe_helper.run_crepe(
        audio_file,
        model_capacity=mont_inputs.crepe_model,
        duration=mont_inputs.crepe_duration,
        offset=mont_inputs.crepe_offset,
        shift_by_half_note=mont_inputs.crepe_shift_by_half_note,
    )

    print_verbose(
        f"Fullmont processing video/audio: offset: {mont_inputs.crepe_offset}s, duration: {mont_inputs.crepe_duration}s, total_duration={video.duration}"
    )
    frame = video.get_frame(audio_pitch_infos[0].timestamp)
    if show_image:
        helper.show_image_with_point(
            frame, [Point.from_coordinates(mont_inputs.sam_input_point)]
        )
    fretboard_mask_result: SAM2MaskResult = get_fretboard_mask_result(
        frame, np.array([mont_inputs.sam_input_point]), show_all_masks=show_image
    )
    vismont_result = run_vismont(
        frame, fretboard_mask_result, mont_inputs, show_image=show_image
    )
    if show_image:
        vismont_result.plot_canny_and_fingertips(exclude_thumb=True)

    guitar1 = Guitar(vismont_result.peaks_vertical)
    if show_image:
        helper.show_image_with_vertical_lines(
            vismont_result.canny,
            guitar1.fret_positions,
            "Final guitar fretboard positions",
        )

    predicted_tabs = []
    for audio_pitch_info in audio_pitch_infos:
        possible_tabs = GuitarTab.possible_tabs(audio_pitch_info.pitch)
        print_verbose(
            f"Fullmont processing: {audio_pitch_info} / Options: {possible_tabs})"
        )
        frame = video.get_frame(audio_pitch_info.timestamp)
        # helper.show_image(frame)
        vismont_result = run_vismont(frame, fretboard_mask_result, mont_inputs)
        if vismont_result == None:
            print_error("Hand was not detected")
            predicted_tabs.append(None)

        else:
            finger_indices = []
            for tip in vismont_result.hand.tips([1, 2, 3]):
                finger_idx = guitar1.get_fret_index(tip.x)
                finger_indices.append(finger_idx)

            print_verbose(possible_tabs, finger_indices)

            for tab in possible_tabs:
                if tab.fret_index in finger_indices:
                    print_verbose(
                        f"Selected tab. pitch={audio_pitch_info.pitch}, tab={tab}"
                    )
                    predicted_tabs.append(tab)

    return predicted_tabs


def test_vismont_on_one_image(file):
    image_bgr = Image.open(file)
    image_rgb = np.array(image_bgr.convert("RGB"))
    # helper.show_image(image_rgb)

    # input_point = np.array([[1600, 200]])  # images/raw/guitar.png
    # input_point = np.array([[2670, 558]])  # sweetchild/screenshot.png
    input_point = np.array([[1200, 230]])  # sweetchild/1.png
    input_label = np.array([1])
    fretboard_mask_result: SAM2MaskResult = get_fretboard_mask_result(
        image_rgb, input_point, input_label, show_all_masks=False
    )
    vismont = run_vismont(image_rgb, fretboard_mask_result)
    # vismont.plot_canny_and_fingertips(exclude_thumb=True)
    lines = helper.run_hough_line(vismont.canny)
    lines = [line for line in lines if helper.is_vertical(line)]
    # helper.show_image_with_lines(vismont.canny, lines, gray=True)
    # edges = helper.dilate(vismont.canny)
    peaks_vertical = helper.find_vertical_sum_peaks(
        vismont.canny, height=2000, distance=25, prominence=500, show_image=True
    )
    helper.show_image_with_vertical_lines(vismont.canny, peaks_vertical)
    return vismont


if __name__ == "__main__":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # TODO: needed?
    # test_vismont_on_one_image("./files/sweetchild/1.png")

    DIR = f"files/sunshine"
    video_file = f"{DIR}/video.mp4"
    audio_file = f"{DIR}/audio.mp3"
    mont_input_file = f"{DIR}/input.json"
    mont_inputs = MontInputs.load_from_json_file(mont_input_file)

    predict_tabs = run_fullmont(video_file, audio_file, mont_inputs, show_image=True)

    tabs_as_str = guitar.tabs2string(predict_tabs)
    with open(f"{DIR}/predicted_tabs_{mont_inputs.crepe_model}.txt", "w") as f:
        for tab in predict_tabs:
            f.write(f"{tab}\n")
    with open(f"{DIR}/predicted_tabs_{mont_inputs.crepe_model}_string.txt", "w") as f:
        f.write(tabs_as_str)

    print_verbose(tabs_as_str)
