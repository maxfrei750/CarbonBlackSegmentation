import torch
from ignite.metrics import Accuracy, ConfusionMatrix, Loss


# TODO: Fix metrics.
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
