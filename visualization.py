import numpy as np
from PIL import Image, ImageFilter


def get_overlay_image(image, ground_truth=None, prediction=None):
    assert not (
        ground_truth is None or prediction is None
    ), "Either ground_truth or prediction need to be specified."

    image = np.array(Image.fromarray(image).convert("RGB"))

    if ground_truth is not None:
        image = _add_overlay(image, ground_truth, (0, 255, 0))

    if prediction is not None:
        image = _add_overlay(image, prediction, (255, 0, 0))

    return image


def _add_overlay(image, mask, color):
    mask = Image.fromarray(mask).convert("L")
    outlines = mask.filter(ImageFilter.FIND_EDGES)
    outlines = np.array(outlines)
    image[np.nonzero(outlines)] = color

    return image
