import numpy as np
from PIL import Image, ImageFilter, ImageOps


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


