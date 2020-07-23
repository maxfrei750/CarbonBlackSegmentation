import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm

from deployment import Segmenter


def demo():
    # Gather paths of images to be segmented.
    data_dir = os.path.join("data", "val", "input")
    image_paths = glob(os.path.join(data_dir, "*.*"))

    # Create a Segmenter object. When segmenting many images, it may be advisable to use a GPU.
    device = "cuda"
    segmenter = Segmenter(device=device)

    # Create an empty list to store the masks.
    masks = []

    # Iterate the image paths.
    for image_path in tqdm(image_paths):
        # Load an image and convert it to an RGB array.
        image = np.asarray(Image.open(image_path).convert("RGB"))

        # Segment the image.
        mask = segmenter.segment_image(image)

        # Store the mask in the list of masks.
        masks.append(mask)


if __name__ == "__main__":
    demo()
