# Static Images: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
# Video Capture: https://mudgalvaibhav.medium.com/real-time-gesture-recognition-using-googles-mediapipe-hands-add-your-own-gestures-tutorial-1-dd7f14169c19

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2, classification_pb2
from mediapipe.python.solutions.hands import Hands
from .helper import *
from typing import List, Tuple
from typing import Optional
from enum import Enum

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class Handedness(Enum):
    LEFT = "Left"
    RIGHT = "Right"


class HandResult:
    NUM_LANDMARKS = len(list(mp_hands.HandLandmark))

    def __init__(
        self,
        handedness: Handedness,
        landmarks: List[Point],
        image_height: int,
        image_width: int,
    ):
        self.handedness = handedness
        self.landmarks = landmarks
        self.image_height = image_height
        self.image_width = image_width

    # returns points of fingertips for thumb, index, ..., pink
    def tips(self, indices: List[int] = None) -> List[Point]:
        all_tips = [
            self.landmarks[mp_hands.HandLandmark.THUMB_TIP],
            self.landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP],
            self.landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            self.landmarks[mp_hands.HandLandmark.RING_FINGER_TIP],
            self.landmarks[mp_hands.HandLandmark.PINKY_TIP],
        ]
        if indices is None:
            return all_tips
        return [all_tips[i] for i in indices]

    # from mediapipe result
    def from_mediapipe_result(
        handedness_mp: classification_pb2.Classification,
        landmarks_normalized_mp: landmark_pb2.NormalizedLandmarkList,
        image_height: int,
        image_width: int,
        perserve_handedness=False,  # mediapipe interprets the image as mirrored image, so use this flag only if original image was already flipped
    ):
        if not perserve_handedness:
            if handedness_mp.label == "Left":
                handedness = Handedness.RIGHT
            else:
                handedness = Handedness.LEFT
        else:
            handedness = Handedness[handedness_mp.label.upper()]

        landmarks: List[Point] = []
        for landmark in landmarks_normalized_mp.landmark:
            landmarks.append(
                Point(landmark.x * image_width, landmark.y * image_height, landmark.z)
            )

        return HandResult(handedness, landmarks, image_height, image_width)

    def rotate_ccw(self, angle_in_degree):
        landmarks = []
        for idx in range(self.NUM_LANDMARKS):
            landmarks.append(
                self.landmarks[idx].rotate_ccw(
                    angle_in_degree, self.image_width // 2, self.image_height // 2
                )
            )
        return HandResult(
            self.handedness, landmarks, self.image_height, self.image_width
        )


def initialize_mp_hands(min_confidence: float = 0.5) -> Hands:
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_confidence,
    )


def run_mp_hands(
    hands: Hands, image: np.ndarray, is_bgr: bool = False
) -> List[HandResult]:
    # image = cv2.flip(image_original, 1) # just flip when we initialize MpHandResult instead
    if is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Print handedness and draw hand landmarks on the image.
    # print_verbose("Handedness:", results.multi_handedness)
    if not results.multi_hand_landmarks:
        return None

    mp_hand_result = []
    for idx in range(len(results.multi_handedness)):
        if VERBOSE:
            hand_landmarks = results.multi_hand_landmarks[idx]
            image_height, image_width, _ = image.shape
            # print_verbose("hand_landmarks:", hand_landmarks)
            # print_verbose(
            #     f"Index finger tip coordinates: (",
            #     f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
            #     f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})",
            # )

        mp_hand_result.append(
            HandResult.from_mediapipe_result(
                results.multi_handedness[idx].classification[0],
                results.multi_hand_landmarks[idx],
                image_height,
                image_width,
            )
        )

    return mp_hand_result


def annotate_mp_hand_result(
    image: np.ndarray, mp_hand_result: HandResult
) -> np.ndarray:
    annotated_image = cv2.flip(image, 1).copy()

    mp_drawing.draw_landmarks(
        annotated_image,
        mp_hand_result.landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )
    return cv2.flip(annotated_image, 1)


if __name__ == "__main__":
    # For static images:
    INPUT_DIR = "./images/raw"
    OUTPUT_DIR = "./images/processed"
    IMAGE_FILES = [
        # "hand.png",
        "guitar.png",  # confidence = 0.1 worked
        # "guitar2.png"  # doesn't work
        # "guitar3.png"
    ]
    for file in IMAGE_FILES:
        image_original = cv2.imread(f"{INPUT_DIR}/{file}")
        base, ext = file.rsplit(".", 1)
        min_confidence = 0.1
        with initialize_mp_hands(min_confidence=min_confidence) as hands:
            mp_hand_results = run_mp_hands(hands, image_original, is_bgr=True)
            if mp_hand_results == None:
                print_verbose(
                    f"hands not detected: file={file}, min_confidence={min_confidence}"
                )
            else:
                for idx, mp_hand_result in enumerate(mp_hand_results):
                    annotated_image = annotate_mp_hand_result(
                        image_original, mp_hand_result
                    )
                    cv2.imwrite(
                        f"{OUTPUT_DIR}/mediapipe_{base}_{idx}.{ext}", annotated_image
                    )
