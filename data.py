import os
import random
from glob import glob

import numpy as np
from PIL import Image

import ignite.distributed as idist
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset
from transforms import get_preprocessing, get_training_augmentation, get_validation_augmentation


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
        while True:
            image, mask, _ = self.get_sample(index)

            if np.any(mask):
                break

        return image, mask

    def get_raw_mask(self, index):
        mask = np.array(Image.open(self.masks[index]).convert("1")).astype("uint8")
        return mask

    def get_raw_image(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        return image

    def __len__(self):
        return len(self.images)

    def get_example_sample(self, index=None):
        if index is None:
            index = random.randint(0, len(self) - 1)

        image, mask, image_vis = self.get_sample(index)

        return image, mask, image_vis

    def get_sample(self, index):
        image = self.get_raw_image(index)
        mask = self.get_raw_mask(index)

        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        image_vis = image

        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        return image, mask, image_vis


def get_train_val_datasets(config):
    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        config["encoder"], config["encoder_weights"]
    )

    dataset_train = SegmentationDataset(
        config["data_path"],
        config["subset_train"],
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    dataset_val = SegmentationDataset(
        config["data_path"],
        config["subset_val"],
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    return dataset_train, dataset_val


def get_dataloaders(config):
    dataset_train, dataset_val = get_train_val_datasets(config)

    # Setup data loader also adapted to distributed config: nccl, gloo, xla-tpu
    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=True,
        drop_last=True,
    )

    dataloader_val = idist.auto_dataloader(
        dataset_val,
        batch_size=2 * config["batch_size"],
        num_workers=config["num_workers"],
        shuffle=False,
    )
    return dataloader_train, dataloader_val


def test():
    import argparse
    from PIL import Image
    from visualization import get_overlay_image

    parser = argparse.ArgumentParser(description="dataset")
    parser.add_argument("data_root", help="data root")
    args = parser.parse_args()

    dataset = SegmentationDataset(args.data_root, "val")

    _, mask, image = dataset.get_example_sample()
    overlay_image = get_overlay_image(image, mask)

    Image.fromarray(overlay_image).show()


if __name__ == "__main__":
    test()
