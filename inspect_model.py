import os

import numpy as np
from PIL import Image

import fire
import torch
from data import SegmentationDataset
from deployment import Segmenter
from ignite.metrics import Accuracy, ConfusionMatrix
from utils import checkpoint_path_to_config
from validation import output_transform_accuracy, output_transform_confusion_matrix
from visualization import get_overlay_image, plot_confusion_matrix


def inspect_model(
    data_root, checkpoint_path, data_subset="val", device="cuda", num_example_predictions=10
):

    output_root = os.path.dirname(checkpoint_path)
    segmenter = get_segmenter(checkpoint_path, device)

    save_confusion_matrix(data_root, output_root, segmenter, data_subset)

    save_example_predictions(
        data_root, output_root, segmenter, data_subset, num_example_predictions
    )


def save_confusion_matrix(data_root, output_root, segmenter, data_subset="val"):
    dataset = SegmentationDataset(data_root, data_subset)

    confusion_matrix_caluclator = ConfusionMatrix(num_classes=2, average="precision")
    accuracy_calculator = Accuracy()

    for image, mask_gt in dataset:
        mask_pred = segmenter.get_raw_prediction(image)
        mask_gt = torch.from_numpy(mask_gt).to(mask_pred.device).unsqueeze(0).unsqueeze(0)

        output = (mask_pred, mask_gt)

        confusion_matrix_caluclator.update(output_transform_confusion_matrix(output))
        accuracy_calculator.update(output_transform_accuracy(output))

    confusion_matrix = confusion_matrix_caluclator.compute()
    accuracy = accuracy_calculator.compute()

    cm_figure = plot_confusion_matrix(confusion_matrix)

    filename_base = f"confusion_matrix_acc={accuracy:.6f}"

    cm_figure.savefig(os.path.join(output_root, filename_base + ".pdf"))
    cm_figure.savefig(os.path.join(output_root, filename_base + ".png"))


def save_example_predictions(
    data_root, output_root, segmenter, data_subset="val", num_predictions=10
):

    example_detection_root = os.path.join(output_root, "example_detections")
    os.makedirs(example_detection_root, exist_ok=True)

    dataset = SegmentationDataset(data_root, data_subset)
    dataset_vis = SegmentationDataset(data_root, data_subset)

    for i in range(num_predictions):
        image_id = np.random.choice(len(dataset))

        image_vis = dataset_vis[image_id][0].astype("uint8")
        image, mask_gt = dataset[image_id]

        mask_pred = segmenter.segment_image(image)
        overlay_image = get_overlay_image(image_vis, prediction=mask_pred, ground_truth=None)

        overlay_image = Image.fromarray(overlay_image)

        filename = f"example_detection_{image_id}.png"

        overlay_image.save(os.path.join(example_detection_root, filename))


def get_segmenter(checkpoint_path, device):
    config = checkpoint_path_to_config(checkpoint_path)
    segmenter = Segmenter(
        architecture=config["architecture"],
        encoder=config["encoder"],
        encoder_weights=config["encoder_weights"],
        device=device,
    )
    return segmenter


if __name__ == "__main__":
    fire.Fire(inspect_model)
