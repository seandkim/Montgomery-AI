import os
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from . import sam2_helper
from . import mediapipe_helper as mp_helper

from .helper import *
from .sam2_helper import SAM2MaskResult


def get_fretboard_mask_result(mask_results: List[SAM2MaskResult]) -> SAM2MaskResult:
    # TODO: implement
    return mask_results[1]


if __name__ == "__main__":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = setup_torch_device()

    file = "./images/raw/guitar.png"
    image = Image.open(file)
    image = np.array(image.convert("RGB"))
    input_point = np.array([[1600, 200]])
    input_label = np.array([1])
    masks_results = sam2_helper.run_sam2(device, image, input_point, input_label)
    fretboard_mask_result: SAM2MaskResult = get_fretboard_mask_result(masks_results)

    min_confidence = 0.1
    with mp_helper.initialize_mp_hands(min_confidence=min_confidence) as hands:
        mp_hand_results = mp_helper.run_mp_hands(hands, image)
        if mp_hand_results == None:
            print_verbose(
                f"hands not detected: file={file}, min_confidence={min_confidence}"
            )
            exit

        for idx, mp_hand_result in enumerate(mp_hand_results):
            annotated_image = mp_helper.annotate_mp_hand_result(image, mp_hand_result)
            # cv2.imwrite(f"{OUTPUT_DIR}/mediapipe_{base}_{idx}.{ext}", annotated_image)

        sam2_helper.show_mask(
            annotated_image,
            fretboard_mask_result,
            point_coords=input_point,
            input_labels=input_label,
            borders=True,
            block=True,
        )
