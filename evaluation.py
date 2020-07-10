import random

import torch
from ignite.metrics import Accuracy, ConfusionMatrix, Loss
from visualization import get_overlay_image


def get_metrics(loss):
    def _output_transform_accuracy(output):
        y_pred, y = output
        y_pred = torch.round(y_pred).flatten(1, -1)

        y = y.flatten(1, -1)

        return y_pred, y

    def _output_transform_confusion_matrix(output):
        y_pred, y = output

        y_pred = y_pred.flatten(1, -1)
        y_pred = torch.stack([y_pred, 1 - y_pred], 1)
        y = y.flatten(1, -1)

        return y_pred, y

    metrics = {
        "loss": Loss(loss),
        "accuracy": Accuracy(_output_transform_accuracy),
        "confusion matrix": ConfusionMatrix(
            num_classes=2, output_transform=_output_transform_confusion_matrix, average="precision"
        ),
    }
    return metrics


class ExamplePredictionLogger:
    def __init__(self, tb_logger, model, device):
        self.model = model
        self.tb_logger = tb_logger
        self.device = device
        self.dataset = None
        self.image_index = None

    def log_visualization(self, dataset, epoch):
        if dataset != self.dataset:
            self.dataset = dataset
            self.image_index = random.randint(0, len(dataset) - 1)

        image, mask_gt, image_vis = dataset.get_example_sample(self.image_index)

        image = torch.from_numpy(image).to(self.device).unsqueeze(0)

        with torch.no_grad():
            mask_pred = self.model(image).cpu().numpy().squeeze().round()

        overlay_image = get_overlay_image(image_vis, mask_gt, mask_pred)

        self.tb_logger.writer.add_image(
            "example prediction", overlay_image, epoch, dataformats="HWC"
        )
