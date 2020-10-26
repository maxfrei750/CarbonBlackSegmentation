import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from sklearn.metrics import ConfusionMatrixDisplay


def get_overlay_image(image, ground_truth=None, prediction=None, alpha=0.1):
    assert not (
        ground_truth is None and prediction is None
    ), "Either ground_truth or prediction need to be specified."

    image = np.array(Image.fromarray(image).convert("RGB"))

    if ground_truth is not None:
        image = _add_overlay(image, ground_truth, (0, 255, 0), alpha=alpha)

    if prediction is not None:
        image = _add_overlay(image, prediction, (255, 0, 0), alpha=alpha)

    return image


def _add_overlay(image, mask, color, alpha=0.1):
    mask = Image.fromarray(mask).convert("L")
    image = Image.fromarray(image)

    outlines = np.array(mask.filter(ImageFilter.FIND_EDGES))

    mask_colored = ImageOps.colorize(mask, (0, 0, 0), color, whitepoint=1)
    overlay_image = np.array(Image.blend(image, mask_colored, alpha))
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


#== VIZ_BINARIES ============================================#
# Function to plot a grid of restults.
# Takes masks and option image_paths argument. If:
#     Only masks, plots B&W binaries. 
#     Both masks and image_paths, uses get_overlay_image above.
# Author: Timothy Sipkens
def viz_binaries(masks, image_paths=''):
    plt.gray() # apply grayscale colormap
    
    no_masks = len(masks) # number of images/masks
    no_col = np.minimum(no_masks, 5) # max. no. of columns is 5
    no_row = int(np.ceil(no_masks / no_col)) # split up over rows
    fig, axs = plt.subplots(no_row, no_col, figsize=(20, 3.5*no_row))
    
    ii = 0 # column
    jj = 0 # row
    kk = 0 # global
    for mask in masks:
        axs[jj,ii].set_xticks([])
        axs[jj,ii].set_yticks([])
        
        if image_paths=='': 
            axs[jj,ii].imshow(mask)
        else: 
            image = np.asarray(Image.open(image_paths[kk]).convert("RGB"))
            axs[jj,ii].imshow(get_overlay_image(image, prediction=mask))
            
        ii = ii + 1
        kk = kk + 1
        if ii==no_col:
            ii = 0
            jj = jj + 1
    
    # for remaining panels, remove axes
    while (ii*jj) <= ((no_col-1)*(no_row-1)):
        axs[jj,ii].axis('off')
        ii = ii + 1
        
    return None
