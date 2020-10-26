import os
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm # used for progress bar in loops

from deployment import Segmenter # local module for segmentation


def demo():
    # Gather paths of images to be segmented.
    data_dir = os.path.join("data/")
    image_paths = glob(os.path.join(data_dir, "*.*"))

    # When segmenting many images, it may be advisable to use a GPU.
    device = "cuda"

    # Perform the segmentation.
    masks = iterate_and_segment_images(image_paths, device)
    
    return masks


def iterate_and_segment_images(image_paths, device="cpu"):

    # Create a Segmenter object.
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
    return masks


if __name__ == "__main__":
    masks = demo()
