import os
from datetime import datetime


def get_time_stamp():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def select_active_gpus(gpu_ids):
    device_string = ", ".join([str(gpu_id) for gpu_id in gpu_ids])

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = device_string
