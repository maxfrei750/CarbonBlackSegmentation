import os

import numpy as np
from PIL import Image

from visualization import get_overlay_image, plot_binary_grid
from wrapper import single_image  # segmenter for single image

if __name__ == "__main__":
    # Define the path to the image.
    image_path = os.path.join("test_images", "201805A_A6_004.png")
    image = Image.open(image_path).convert("RGB")  # open image
    image = np.asarray(image)  # convert image to numpy array

    # Apply segmenter to default test images.
    print("Classifying image...")
    mask = single_image(image_path)
    print("Complete.")

    # Display result (two different ways).
    overlay_image = get_overlay_image(image, prediction=mask, alpha=0)
    Image.fromarray(overlay_image).show()  # show the overlay generated above (external)
    plot_binary_grid([mask])  # show binary image in console, note the []
