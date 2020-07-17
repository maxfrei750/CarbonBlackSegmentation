import os
import warnings
from glob import glob

import fire


def check_samples(
    base_directory,
    inspection_folder="inspection",
    input_folder="input",
    ground_truth_folder="ground_truth",
):
    input_image_paths = glob(os.path.join(base_directory, input_folder, "*.*"))

    for input_image_path in input_image_paths:
        filename = os.path.basename(input_image_path)
        ground_truth_path = os.path.join(base_directory, ground_truth_folder, filename)
        inspection_image_path = os.path.join(base_directory, inspection_folder, filename)

        if not os.path.exists(inspection_image_path):
            os.remove(input_image_path)
            os.remove(ground_truth_path)
            warnings.warn(f"Missing inspection image: {filename}")


if __name__ == "__main__":
    fire.Fire(check_samples)
