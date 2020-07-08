import numpy as np
from PIL import Image, ImageFilter


def get_overlay_image(input_image, mask, outline_color=(255, 0, 0)):
    input_image = Image.fromarray(input_image).convert("L").convert("RGB")
    mask = Image.fromarray(mask).convert("L")
    mask_outlines = mask.filter(ImageFilter.FIND_EDGES)
    mask_outlines = np.array(mask_outlines)
    overlay_image = np.array(input_image)
    overlay_image[np.nonzero(mask_outlines)] = outline_color
    return overlay_image
