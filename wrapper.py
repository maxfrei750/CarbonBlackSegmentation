"""
This module contains a series of wrappers, that take an input
in the form of image paths and output segmented masks.
Various renditions are available. Demonstration of the use of
these wrappers is provided in the scripts in the demo folder.
"""

import os  # used in generating file paths
from glob import glob

import numpy as np
from PIL import Image
from tqdm import tqdm  # used for progress bar in loops

from deployment import Segmenter  # local module for segmentation


def single_image(image_path="", device="cpu"):
    """Simple wrapper for segmentation of a single image.

    Contributors: @tsipkens, @maxfrei750, Oct/2020

    :param image_paths: single of image path
    :param device: e.g., "cpu" or "cuda"
    """

    # Load image.
    # By default, if image_path="", use the test image 201805A_A6_004.png.
    if image_path == "":
        image_path = os.path.join("test_images", "201805A_A6_004.png")

    image = Image.open(image_path).convert("RGB")  # open image
    image = np.asarray(image)  # convert image to numpy array

    segmenter = Segmenter(device=device)  # create a Segmenter object
    mask = segmenter.segment_image(image)  # segment an image

    return mask


def multi_image(image_paths="", device="cuda"):
    """Simple wrapper for segmentation of multiple image.
    When segmenting many images, it may be advisable to use a GPU.
    Thus, device="cuda" by default.

    Contributors: @tsipkens, @maxfrei750, Oct/2020

    :param image_paths: single of image path
    :param device: e.g., "cpu" or "cuda"
    """

    # Gather paths of images to be segmented.
    # By default, if image_paths="" or image_paths=[],
    # use images in test_images folder.
    if (image_paths == "") or (image_paths == []):
        data_dir = os.path.join("test_images")
        image_paths = glob(os.path.join(data_dir, "*.*"))

    # Perform the segmentation.
    segmenter = Segmenter(device=device)  # create a Segmenter object
    masks = []  # create an empty list to store the masks
    for image_path in tqdm(image_paths):  # iterate the image paths

        # Load an image and convert it to an RGB array.
        image = np.asarray(Image.open(image_path).convert("RGB"))

        mask = segmenter.segment_image(image)  # segment the image
        masks.append(mask)  # store the mask in the list of masks.

    return masks
