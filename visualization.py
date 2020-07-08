import numpy as np
from PIL import Image, ImageFilter

# from skimage import img_as_ubyte
# from skimage.exposure import rescale_intensity


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


# def get_logging_image(prediction, image, outline_color=(255, 0, 0)):
#     # Partially based on:
#     #   https://stackoverflow.com/questions/57576686/how-to-overlay-segmented-image-on-top-of-main-image-in-python
#
#     mask = prediction.argmax(dim=1).squeeze().cpu().numpy().astype("uint8") * 255
#
#     image = image.squeeze().cpu().numpy()
#     image = np.rollaxis(image, 0, 3)
#
#     image = rescale_intensity(image)
#     image = img_as_ubyte(image)
#
#     logging_image = get_overlay_image(image, mask, outline_color)
#
#     return logging_image
