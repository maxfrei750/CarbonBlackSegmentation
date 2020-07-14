import os
import warnings
from glob import glob

import fire


def check_samples(base_directory, input_folder="input", ground_truth_folder="ground_truth"):

    ground_truth_paths = glob(os.path.join(base_directory, ground_truth_folder, "*.*"))

    for ground_truth_path in ground_truth_paths:
        filename = os.path.basename(ground_truth_path)
        input_image_path = os.path.join(base_directory, input_folder, filename)

        if not os.path.exists(input_image_path):
            os.remove(ground_truth_path)
            warnings.warn(f"Missing input image: {filename}")


if __name__ == "__main__":
    fire.Fire(check_samples)
