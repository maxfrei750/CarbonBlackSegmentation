import os
import random
import shutil
from glob import glob

import fire


def split_data(
    base_directory,
    input_image_folder_in="input",
    ground_truth_folder_in="ground_truth",
    val_ratio=0.15,
    random_seed=0,
):
    random.seed(random_seed)

    input_image_folder_out_val = os.path.join(base_directory, "val", input_image_folder_in)
    ground_truth_folder_out_val = os.path.join(base_directory, "val", ground_truth_folder_in)

    os.makedirs(input_image_folder_out_val, exist_ok=True)
    os.makedirs(ground_truth_folder_out_val, exist_ok=True)

    input_image_paths = glob(os.path.join(base_directory, input_image_folder_in, "*.*"))

    num_val_samples = round(len(input_image_paths) * val_ratio)
    val_image_paths_in = random.sample(input_image_paths, num_val_samples)

    for val_image_path_in in val_image_paths_in:
        filename = os.path.basename(val_image_path_in)

        ground_truth_path_in = os.path.join(base_directory, ground_truth_folder_in, filename)
        ground_truth_path_out = os.path.join(base_directory, ground_truth_folder_out_val, filename)

        val_image_path_out = os.path.join(base_directory, input_image_folder_out_val, filename)

        shutil.move(val_image_path_in, val_image_path_out)
        shutil.move(ground_truth_path_in, ground_truth_path_out)

    shutil.move(
        os.path.join(base_directory, input_image_folder_in),
        os.path.join(base_directory, "train", input_image_folder_in),
    )
    shutil.move(
        os.path.join(base_directory, ground_truth_folder_in),
        os.path.join(base_directory, "train", ground_truth_folder_in),
    )


if __name__ == "__main__":
    fire.Fire(split_data)
