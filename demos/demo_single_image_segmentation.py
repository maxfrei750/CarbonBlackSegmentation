import os

import numpy as np
from PIL import Image

from deployment import Segmenter  # used for progress bar in loops
from visualization import get_overlay_image  # used to show results


def demo():
    # Select a device to use (either "cpu" or "cuda").
    device = "cpu"

    # Load image.
    image_path = os.path.join("..", "test_images", "201805A_A6_004.png")
    image = Image.open(image_path).convert("RGB")

    # Convert image to numpy array.
    image = np.asarray(image)

    # Create a Segmenter object.
    segmenter = Segmenter(device=device)

    # Segment an image.
    mask = segmenter.segment_image(image)

    # Display result.
    overlay_image = get_overlay_image(image, prediction=mask, alpha=0)
    Image.fromarray(overlay_image).show()


if __name__ == "__main__":
    demo()
