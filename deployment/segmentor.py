import os
from glob import glob

import numpy as np
from PIL import Image

import segmentation_models_pytorch as smp
import torch
from ignite.handlers import Checkpoint
from modeling import get_model
from transforms import get_preprocessing, get_validation_augmentation
from visualization import get_overlay_image


class Segmentor:
    def __init__(
        self,
        architecture="FPN",
        encoder="resnet50",
        encoder_weights="imagenet",
        checkpoint_path=None,
        device="cpu",
    ):

        self.config = {
            "architecture": architecture,
            "encoder": encoder,
            "encoder_weights": encoder_weights,
        }

        if checkpoint_path is None:
            self.checkpoint_path = os.path.join(
                os.path.dirname(__file__), "FPN-resnet50-imagenet.pt"
            )
        else:
            self.checkpoint_path = checkpoint_path

        self.device = device

        self.model = self.load_model()

        self.preprocessing_fn = get_preprocessing(
            smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        )
        self.augmentation_fn = get_validation_augmentation()

    def load_model(self):
        model = get_model(self.config)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
        model.eval()

        return model

    def segment_image(self, image):
        image = self.augmentation_fn(image=image)["image"]
        image = self.preprocessing_fn(image=image)["image"]

        image = torch.from_numpy(image).to(self.device).unsqueeze(0)

        with torch.no_grad():
            mask = self.model(image).squeeze().cpu().numpy().round().astype(bool)

        return mask


if __name__ == "__main__":
    import random

    data_dir = os.path.join("..", "data", "val", "input")
    test_image_path = random.sample(glob(os.path.join(data_dir, "*.*")), 1)[0]
    test_image = np.asarray(Image.open(test_image_path).convert("RGB"))
    segmentor = Segmentor()
    mask = segmentor.segment_image(test_image)

    overlay_image = get_overlay_image(test_image, prediction=mask, alpha=0)
    Image.fromarray(overlay_image).show()
