import fire
from deployment import Segmenter
from utils import checkpoint_path_to_config


def convert_model_to_onnx(pt_file_path=None, onnx_file_path=None):
    """Function to convert pt model checkpoints to onnx files.

    :param pt_file_path: Checkpoint file path. If None, uses the default model (deployment/FPN-resnet50-imagenet.pt).
    :param onnx_file_path: Output onnx file path. If None, uses pt_file_path, but replaces .pt with .onnx
    """

    if pt_file_path is not None:
        config = checkpoint_path_to_config(pt_file_path)

        segmenter = Segmenter(
            architecture=config["architecture"],
            encoder=config["encoder"],
            encoder_weights=config["encoder_weights"],
        )
    else:
        segmenter = Segmenter()

    segmenter.save_model_as_onnx(onnx_file_path)


if __name__ == "__main__":
    fire.Fire(convert_model_to_onnx)
