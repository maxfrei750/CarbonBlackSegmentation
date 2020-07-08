import os
from glob import glob

import numpy as np
from PIL import Image

from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    """Segmentation Dataset. Read images and masks and apply augmentation and preprocessing transformations.
    """

    def __init__(
        self,
        root,
        image_set,
        image_subfolder="input",
        mask_subfolder="ground_truth",
        augmentation=None,
        preprocessing=None,
    ):

        self.root = root
        self.image_set = image_set
        self.image_subfolder = image_subfolder
        self.mask_subfolder = mask_subfolder
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        self._gather_paths()

    def _gather_paths(self):
        subset_root = os.path.join(self.root, self.image_set)

        if not os.path.isdir(subset_root):
            raise RuntimeError(f"Dataset not found: {self.image_set}")

        image_dir = os.path.join(subset_root, self.image_subfolder)
        mask_dir = os.path.join(subset_root, self.mask_subfolder)

        if not (os.path.isdir(image_dir) and os.path.isdir(mask_dir)):
            raise RuntimeError(f"Dataset corrupted: {self.image_set}")

        self.images = sorted(glob(os.path.join(image_dir, "*.*")))
        self.masks = sorted(glob(os.path.join(mask_dir, "*.*")))

        if len(self.images) != len(self.masks):
            raise RuntimeError(f"Unequal number of images and masks for dataset {self.image_set}.")

    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        mask = np.array(Image.open(self.masks[index]).convert("1")).astype("uint8")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask

    def __len__(self):
        return len(self.images)


def test():
    import argparse
    from PIL import Image
    import random
    from visualization import get_overlay_image

    parser = argparse.ArgumentParser(description="dataset")
    parser.add_argument("data_root", help="data root")
    args = parser.parse_args()

    dataset = SegmentationDataset(args.data_root, "test")

    sample_id = random.randint(0, len(dataset))
    image, mask = dataset[sample_id]
    overlay_image = get_overlay_image(image, mask)

    Image.fromarray(overlay_image).show()


if __name__ == "__main__":
    test()
