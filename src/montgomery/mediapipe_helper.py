# Static Images: https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/hands.md
# Video Capture: https://mudgalvaibhav.medium.com/real-time-gesture-recognition-using-googles-mediapipe-hands-add-your-own-gestures-tutorial-1-dd7f14169c19

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2, classification_pb2
from mediapipe.python.solutions.hands import Hands
from .helper import *
from typing import List
from typing import Optional

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


class MpHandResult:
    def __init__(
        self,
        landmarks=landmark_pb2.LandmarkList,
        landmarks_normalized=landmark_pb2.NormalizedLandmarkList,
        handedness=classification_pb2.ClassificationList,
    ):
        self.landmarks = landmarks
        self.landmarks_normalized = landmarks_normalized
        self.handedness = handedness


def initialize_mp_hands(min_confidence: float = 0.5) -> Hands:
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=min_confidence,
    )


def run_mp_hands(
    hands: Hands, image_original: np.ndarray, is_bgr: bool = False
) -> Optional[List[MpHandResult]]:
    image = cv2.flip(image_original, 1)  # flip y-axis for correct handedness
    if is_bgr:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Print handedness and draw hand landmarks on the image.
    print_verbose("Handedness:", results.multi_handedness)
    if not results.multi_hand_landmarks:
        return None

    mp_hand_result = []
    for idx in range(len(results.multi_handedness)):
        if VERBOSE:
            hand_landmarks = results.multi_hand_landmarks[idx]
            image_height, image_width, _ = image_original.shape
            print_verbose("hand_landmarks:", hand_landmarks)
            print_verbose(
                f"Index finger tip coordinates: (",
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, "
                f"{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})",
            )

        mp_hand_result.append(
            MpHandResult(
                results.multi_hand_world_landmarks[idx],
                results.multi_hand_landmarks[idx],
                results.multi_handedness[idx],
            )
        )

    return mp_hand_result


def annotate_mp_hand_result(
    image: np.ndarray, mp_hand_result: MpHandResult
) -> np.ndarray:
    annotated_image = cv2.flip(image, 1).copy()

    mp_drawing.draw_landmarks(
        annotated_image,
        mp_hand_result.landmarks_normalized,
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
