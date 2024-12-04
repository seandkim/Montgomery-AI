
import os
import numpy as nps
import matplotlib.pyplot as plt
from PIL import Image

from .helper import *
from .colab_helper import *

if __name__ == "__main__":
  os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
  device = setup_torch_device()

  image = Image.open('./images/raw/guitar.png')
  image = np.array(image.convert("RGB"))
  input_point = np.array([[1600, 200]])
  input_label = np.array([1])
  sam2result = run_sam2(device, image, input_point, input_label)

  show_masks(image, sam2result.masks, sam2result.scores, point_coords=input_point, input_labels=input_label, borders=True)

