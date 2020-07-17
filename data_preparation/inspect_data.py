import os
import warnings
from glob import glob

import numpy as np
from PIL import Image, ImageFilter
from tqdm import tqdm

import fire


def inspect_data(
    base_directory,
    input_folder="input",
    ground_truth_folder="ground_truth",
    inspection_folder="inspection",
):

    os.makedirs(os.path.join(base_directory, inspection_folder), exist_ok=True)

    input_image_paths = glob(os.path.join(base_directory, input_folder, "*.*"))

    for input_image_path in tqdm(input_image_paths):
        filename = os.path.basename(input_image_path)
        ground_truth_path = os.path.join(base_directory, ground_truth_folder, filename)
        inspection_image_path = os.path.join(base_directory, inspection_folder, filename)

        if os.path.exists(inspection_image_path):
            continue

        input_image = Image.open(input_image_path)

        try:
            ground_truth = Image.open(ground_truth_path)
        except FileNotFoundError:
            warnings.warn(f"Missing ground truth: {filename}")
            continue

        inspection_image = get_inspection_image(ground_truth, input_image)

        inspection_image.save(inspection_image_path)


def get_inspection_image(ground_truth, input_image, outline_color=(255, 0, 0)):
    # Partially based on:
    #   https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
    input_image = input_image.convert("L").convert("RGB")
    ground_truth = ground_truth.convert("L")

    ground_truth_outlines = ground_truth.filter(ImageFilter.FIND_EDGES)
    ground_truth_outlines = np.array(ground_truth_outlines)

    inspection_image = np.array(input_image)
    inspection_image[np.nonzero(ground_truth_outlines)] = outline_color
    inspection_image = Image.fromarray(inspection_image)

    return inspection_image


if __name__ == "__main__":
    fire.Fire(inspect_data)
