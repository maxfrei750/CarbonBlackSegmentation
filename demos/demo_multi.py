from visualization import plot_binary_grid  # used to show results
from wrapper import multi_image  # import segmenter for multiple images

if __name__ == "__main__":
    # Apply segmenter to default test images.
    print("Classifying images...")
    masks = multi_image()  # uses default test image
    print("Complete.")

    # Show results.
    plot_binary_grid(masks)
