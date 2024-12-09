# https://colab.research.google.com/github/facebookresearch/sam2/blob/main/notebooks/image_predictor_example.ipynb#scrollTo=226df881

from typing import List, Optional
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from .helper import *

sam2_checkpoint = "src/models/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


class SAM2MaskResult:
    def __init__(self, mask: np.ndarray, score: np.float32, logit: np.ndarray):
        self.mask: np.ndarray = mask
        self.score: np.float32 = score
        self.logit: np.ndarray = logit

    def __repr__(self):
        return f"SAM2Result(masks_shape={self.mask.shape}, scores={self.score:.2f}, logits_shape={self.logit.shape})"

    # e.g. 3 o'clock needle => 0, 12 o'clock needle => 90
    def get_angle_from_positive_x_axis(self) -> np.float64:
        return get_angle_from_positive_x_axis(self.mask)

    def rotate_ccw(self, orientation):
        return SAM2MaskResult(
            rotate_ccw(self.mask, orientation), self.score, self.logit
        )

    def apply_to_image(self, image: np.ndarray) -> np.ndarray:
        masked = np.copy(image)
        masked[self.mask == 0] = np.array([0, 0, 0])
        return masked


# region show function
def show_image_with_input_point(
    image: np.array,
    input_points: Optional[np.ndarray] = None,
    input_label: Optional[np.ndarray] = None,
):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    if input_points is not None and input_label is not None:
        show_points(input_points, input_label, plt.gca())
    plt.axis("on")
    plt.show()


def show_mask_helper(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [
            cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours
        ]
        mask_image = cv2.drawContours(
            mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
        )
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


def show_mask(
    image: np.array,
    mask_result: SAM2MaskResult,
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
    block=False,
):
    mask = mask_result.mask
    score = mask_result.score

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask_helper(mask, plt.gca(), borders=borders)
    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())
    if box_coords is not None:
        # boxes
        show_box(box_coords, plt.gca())

    plt.title(f"Score: {score:.3f}", fontsize=18)
    plt.axis("off")
    plt.show(block=block)


def show_masks(
    image: np.array,
    mask_results: List[SAM2MaskResult],
    point_coords=None,
    box_coords=None,
    input_labels=None,
    borders=True,
    block=False,
):
    for i in range(len(mask_results)):
        mask = mask_results[i].mask
        score = mask_results[i].score

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask_helper(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(mask_results) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis("off")
        plt.show(block=block)


# endregion


# Returns result in the decreasing score
def run_sam2(
    device: torch.device,
    image: np.ndarray,
    input_point: np.ndarray,
    input_label: np.ndarray,
    min_score_threshold: float = 0.1,
) -> List[SAM2MaskResult]:
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)

    # print_verbose(
    #     predictor._features["image_embed"].shape,
    #     predictor._features["image_embed"][-1].shape,
    # )

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    mask_results = []
    for idx in range(len(masks)):
        if scores[idx] > min_score_threshold:
            mask = np.where(masks[idx] == 1, 255, 0).astype(
                np.uint8
            )  # sam2 returns 0,1
            mask_results.append(SAM2MaskResult(mask, scores[idx], logits[idx]))

    return mask_results


if __name__ == "__main__":
    device = setup_torch_device()

    image = Image.open("./images/raw/guitar.png")
    image = np.array(image.convert("RGB"))
    input_point = np.array([[1600, 200]])
    input_label = np.array([1])
    sam2result = run_sam2(device, image, input_point, input_label)

    show_masks(
        image,
        sam2result.masks,
        sam2result.scores,
        point_coords=input_point,
        input_labels=input_label,
        borders=True,
    )
