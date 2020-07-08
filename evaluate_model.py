import os

import numpy as np
from PIL import Image

import segmentation_models_pytorch as smp
import torch
from data import SegmentationDataset
from ignite.handlers import Checkpoint
from modeling import get_model
from transforms import get_preprocessing, get_validation_augmentation
from visualization import get_overlay_image


def checkpoint_path_to_config(checkpoint_path):
    keys = ["architecture", "encoder", "encoder_weights"]
    values = os.path.basename(os.path.dirname(checkpoint_path)).split("_")[0].split("-")
    return dict(zip(keys, values))


def get_example_predictions():
    checkpoint_path = "output/FPN-resnet50-imagenet_backend-None-1_20200709-005112/best_model_9_validation_accuracy=0.0301.pt"
    data_root = "/home/frei/sciebo/Dissertation/Referenzdaten/UBC_TEM"
    subset = "val"

    config = checkpoint_path_to_config(checkpoint_path)

    model = get_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
    model.eval()

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config["encoder"], config["encoder_weights"]
    )

    dataset = SegmentationDataset(
        data_root,
        subset,
        preprocessing=get_preprocessing(preprocessing_fn),
        augmentation=get_validation_augmentation(),
    )

    dataset_vis = SegmentationDataset(data_root, subset)

    for i in range(3):
        n = np.random.choice(len(dataset))

        image_vis = dataset_vis[n][0].astype("uint8")
        image, gt_mask = dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to("cuda").unsqueeze(0)

        with torch.no_grad():
            pr_mask = model(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        overlay_image = get_overlay_image(image_vis, prediction=pr_mask, ground_truth=gt_mask)
        Image.fromarray(overlay_image).show()


if __name__ == "__main__":
    get_example_predictions()
