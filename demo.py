import numpy as np
from PIL import Image

from deployment import Segmentor
from visualization import get_overlay_image


def demo():
    # Select a device to use (either "cpu" or "cuda").
    device = "cpu"

    # Load image.
    image_path = "test_image.png"
    image = Image.open(image_path).convert("RGB")

    # Convert image to numpy array.
    image = np.asarray(image)

    # Create a Segmentor object.
    segmentor = Segmentor(device=device)

    # Segment an image.
    mask = segmentor.segment_image(image)

    # Display result.
    overlay_image = get_overlay_image(image, prediction=mask, alpha=0)
    Image.fromarray(overlay_image).show()


if __name__ == "__main__":
    demo()