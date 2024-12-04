import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from .helper import *
from .colab_helper import *
from .mediapipe_helper import *

if __name__ == "__main__":
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    device = setup_torch_device()

    file = "./images/raw/guitar.png"
    image = Image.open(file)
    image = np.array(image.convert("RGB"))
    input_point = np.array([[1600, 200]])
    input_label = np.array([1])
    sam2result = run_sam2(device, image, input_point, input_label)

    min_confidence = 0.1
    with initialize_mp_hands(min_confidence=min_confidence) as hands:
        mp_hand_results = run_mp_hands(hands, image)
        if mp_hand_results == None:
            print_verbose(
                f"hands not detected: file={file}, min_confidence={min_confidence}"
            )
            exit

        for idx, mp_hand_result in enumerate(mp_hand_results):
            annotated_image = annotate_mp_hand_result(image, mp_hand_result)
            # cv2.imwrite(f"{OUTPUT_DIR}/mediapipe_{base}_{idx}.{ext}", annotated_image)

        show_masks(
            annotated_image,
            sam2result.masks,
            sam2result.scores,
            point_coords=input_point,
            input_labels=input_label,
            borders=True,
            block=True,
        )
