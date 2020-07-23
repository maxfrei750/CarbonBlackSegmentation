import os

import numpy as np
from PIL import Image

import fire
from data import SegmentationDataset
from deployment import Segmenter
from visualization import get_overlay_image


def checkpoint_path_to_config(checkpoint_path):
    keys = ["architecture", "encoder", "encoder_weights"]
    values = os.path.basename(os.path.dirname(checkpoint_path)).split("_")[0].split("-")
    return dict(zip(keys, values))


def get_example_predictions(
    data_root, checkpoint_path, data_subset="val", device="cuda", num_predictions=3
):
    config = checkpoint_path_to_config(checkpoint_path)

    segmenter = Segmenter(
        architecture=config["architecture"],
        encoder=config["encoder"],
        encoder_weights=config["encoder_weights"],
        device=device,
    )

    dataset = SegmentationDataset(data_root, data_subset)
    dataset_vis = SegmentationDataset(data_root, data_subset)

    for i in range(num_predictions):
        image_id = np.random.choice(len(dataset))

        image_vis = dataset_vis[image_id][0].astype("uint8")
        image, mask_gt = dataset[image_id]

        mask_pred = segmenter.segment_image(image)
        overlay_image = get_overlay_image(image_vis, prediction=mask_pred, ground_truth=mask_gt)
        Image.fromarray(overlay_image).show()


if __name__ == "__main__":
    fire.Fire(get_example_predictions)
