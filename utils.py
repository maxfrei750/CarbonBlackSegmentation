import os
from datetime import datetime

import requests
from ignite.utils import setup_logger
from tqdm import tqdm

CHECKPOINT_URL_BASE = (
    "https://github.com/maxfrei750/CarbonBlackSegmentation/releases/download/v1.0/"
)


def get_time_stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def select_active_gpus(gpu_ids):
    device_string = ", ".join([str(gpu_id) for gpu_id in gpu_ids])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_string


def download_checkpoint(checkpoint_path):
    """Based on:
    https://stackoverflow.com/questions/37573483/progress-bar-while-download-file-over-http-with-requests
    """
    checkpoint_filename = os.path.basename(checkpoint_path)

    expected_checkpoint_filenames = ["FPN-resnet50-imagenet.pt"]

    assert (
        checkpoint_filename in expected_checkpoint_filenames
    ), f"Expected checkpoint file name to be in {checkpoint_filename}."

    url = os.path.join(CHECKPOINT_URL_BASE, checkpoint_filename)
    request_stream = requests.get(url, stream=True)

    # Total size in bytes.
    total_size = int(request_stream.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte

    logger = setup_logger()
    logger.info(f"Downloading checkpoint file from {url}...")

    # print(f"Downloading checkpoint file from {url}...")
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(checkpoint_path, "wb") as file:
        for data in request_stream.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size != 0 and progress_bar.n != total_size:
        raise RuntimeError("Error while downloading checkpoint file.")


def checkpoint_path_to_config(checkpoint_path):
    keys = ["architecture", "encoder", "encoder_weights"]
    values = os.path.basename(os.path.dirname(checkpoint_path)).split("_")[0].split("-")
    return dict(zip(keys, values))
