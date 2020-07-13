import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay


def get_overlay_image(image, ground_truth=None, prediction=None):
    assert not (
        ground_truth is None and prediction is None
    ), "Either ground_truth or prediction need to be specified."

    image = np.array(Image.fromarray(image).convert("RGB"))

    if ground_truth is not None:
        image = _add_overlay(image, ground_truth, (0, 255, 0))

    if prediction is not None:
        image = _add_overlay(image, prediction, (255, 0, 0))

    return image


def _add_overlay(image, mask, color):
    mask = Image.fromarray(mask).convert("L")
    image = Image.fromarray(image)

    outlines = np.array(mask.filter(ImageFilter.FIND_EDGES))

    mask_colored = ImageOps.colorize(mask, (0, 0, 0), color, whitepoint=1)
    overlay_image = np.array(Image.blend(image, mask_colored, 0.1))
    overlay_image[np.nonzero(outlines)] = color

    return overlay_image


def plot_confusion_matrix(confusion_matrix_data):
    # Adjust data format to ConfusionMatrixDisplay.
    confusion_matrix_data = np.fliplr(confusion_matrix_data).T

    display_labels = ["background", "agglomerate"]

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix_data, display_labels=display_labels
    )

    display.plot(cmap="Blues")

    for im in display.ax_.get_images():
        upper_limit = 1 if confusion_matrix_data.max() <= 1 else None
        im.set_clim(0, upper_limit)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    plt.tight_layout()

    return plt.gcf()
