import os
from glob import glob

import numpy as np
import segmentation_models_pytorch as smp
import torch
from ignite.handlers import Checkpoint
from PIL import Image

from modeling import get_model
from transforms import get_preprocessing, get_validation_augmentation
from utils import download_checkpoint
from visualization import get_overlay_image


class Segmenter:
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
            "device": device,
        }

        if checkpoint_path is None:
            checkpoint_path = os.path.join(os.path.dirname(__file__), "FPN-resnet50-imagenet.pt")

            if not os.path.exists(checkpoint_path):
                download_checkpoint(checkpoint_path)

        self.checkpoint_path = checkpoint_path

        self.device = device

        self.model = self._load_model()

        self.preprocessing_fn = get_preprocessing(
            smp.encoders.get_preprocessing_fn(encoder, encoder_weights)
        )
        self.augmentation_fn = get_validation_augmentation()

    def _load_model(self):
        model = get_model(self.config)
        model.to(self.device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        Checkpoint.load_objects(to_load={"model": model}, checkpoint=checkpoint)
        model.eval()

        return model

    def segment_image(self, image):
        prediction = self.get_raw_prediction(image)
        return prediction.squeeze().cpu().numpy().round().astype(bool)

    def get_raw_prediction(self, image):
        original_shape = image.shape[:2]
        image = self._prepare_input_image(image)

        with torch.no_grad():
            prediction = self.model(image)

        prediction = Segmenter._postprocess_output(prediction, original_shape)

        return prediction

    @staticmethod
    def _postprocess_output(prediction, original_shape):
        # Crop padding that may have been added during the preprocessing.
        original_height, original_width = original_shape
        height, width = prediction.shape[2:]

        padding_top = int((height - original_height) / 2.0)
        padding_bottom = height - original_height - padding_top

        padding_left = int((width - original_width) / 2.0)
        padding_right = width - original_width - padding_left

        return prediction[
            :, :, padding_top : height - padding_bottom, padding_left : width - padding_right
        ]

    def _prepare_input_image(self, image):
        image = self.augmentation_fn(image=image)["image"]
        image = self.preprocessing_fn(image=image)["image"]
        image = torch.from_numpy(image).to(self.device).unsqueeze(0)
        return image

    def save_model_as_onnx(self, onnx_file_path=None):

        if onnx_file_path is None:
            onnx_file_path = os.path.splitext(self.checkpoint_path)[0] + ".onnx"

        batch_size = 1
        width = 2240
        height = 1952

        input_shape = (batch_size, 3, height, width)

        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }

        input_names = ["input"]
        output_names = ["output"]

        inputs = torch.ones(*input_shape)

        torch.onnx.export(
            self.model,
            inputs,
            onnx_file_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=11,
            verbose=False,
        )


if __name__ == "__main__":
    import random

    data_dir = os.path.join("..", "data", "val", "input")
    test_image_path = random.sample(glob(os.path.join(data_dir, "*.*")), 1)[0]
    test_image = np.asarray(Image.open(test_image_path).convert("RGB"))
    segmenter = Segmenter()
    mask = segmenter.segment_image(test_image)

    overlay_image = get_overlay_image(test_image, prediction=mask, alpha=0)
    Image.fromarray(overlay_image).show()
