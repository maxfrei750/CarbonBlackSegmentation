import numpy as np
from PIL import Image

import segmentation_models_pytorch as smp
import torch
from data import SegmentationDataset
from transforms import get_preprocessing, get_validation_augmentation
from visualization import get_overlay_image


def get_example_predictions():
    best_model = torch.load("output/2020-07-08_16-12-59/best_model.pth")

    preprocessing_fn = smp.encoders.get_preprocessing_fn("resnet50", "imagenet")

    dataset = SegmentationDataset(
        "/home/frei/sciebo/Dissertation/Referenzdaten/UBC_TEM",
        "test",
        preprocessing=get_preprocessing(preprocessing_fn),
        augmentation=get_validation_augmentation(),
    )

    dataset_vis = SegmentationDataset(
        "/home/frei/sciebo/Dissertation/Referenzdaten/UBC_TEM", "test"
    )

    for i in range(3):
        n = np.random.choice(len(dataset))

        image_vis = dataset_vis[n][0].astype("uint8")
        image, gt_mask = dataset[n]

        gt_mask = gt_mask.squeeze()

        x_tensor = torch.from_numpy(image).to("cuda").unsqueeze(0)
        pr_mask = best_model.predict(x_tensor)
        pr_mask = pr_mask.squeeze().cpu().numpy().round()

        overlay_image = get_overlay_image(image_vis, prediction=pr_mask, ground_truth=gt_mask)
        Image.fromarray(overlay_image).show()


if __name__ == "__main__":
    get_example_predictions()
