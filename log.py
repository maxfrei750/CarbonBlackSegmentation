import random

import ignite
import torch
from ignite import distributed as idist
from ignite.handlers import DiskSaver
from visualization import get_overlay_image, plot_confusion_matrix


def setup_trains_logging(config):
    if config["with_trains"]:
        from trains import Task

        task = Task.init("Carbon Black Semantic Segmentation Training", config["task_name"])
        task.connect_configuration(config)

        # Log hyper parameters
        hyper_parameters = list(config.keys())
        task.connect({k: config[k] for k in hyper_parameters})


def log_confusion_matrix(tb_logger, epoch, data_subset, metrics):
    figure = plot_confusion_matrix(metrics["confusion matrix"].cpu().numpy())
    tb_logger.writer.add_figure(f"confusion matrix/{data_subset}", figure, global_step=epoch)


def log_metrics(logger, epoch, elapsed, tag, metrics):
    metric_string = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(f"\nEpoch {epoch} - elapsed: {elapsed} - {tag} metrics:\n {metric_string}")


def log_basic_info(logger, config):
    logger.info(
        f"Train {config['architecture']}_{config['encoder']} on Carbon Black Semantic Segmentation"
    )
    # noinspection PyUnresolvedReferences
    logger.info(f"- PyTorch version: {torch.__version__}")
    logger.info(f"- Ignite version: {ignite.__version__}")

    logger.info("\n")
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"\t{key}: {value}")
    logger.info("\n")

    if idist.get_world_size() > 1:
        logger.info("\nDistributed setting:")
        logger.info(f"\tbackend: {idist.backend()}")
        logger.info(f"\tworld size: {idist.get_world_size()}")
        logger.info("\n")


def get_save_handler(config):
    if config["with_trains"]:
        from ignite.contrib.handlers.trains_logger import TrainsSaver

        return TrainsSaver(dirname=config["output_path"])

    return DiskSaver(config["output_path"], require_empty=False)


class ExamplePredictionLogger:
    def __init__(self, tb_logger, model, device):
        self.model = model
        self.tb_logger = tb_logger
        self.device = device
        self.dataset = None
        self.image_index = None

    def log_visualization(self, dataset, epoch):
        if dataset != self.dataset:
            self.dataset = dataset
            self.image_index = random.randint(0, len(dataset) - 1)

        image, mask_gt, image_vis = dataset.get_example_sample(self.image_index)

        image = torch.from_numpy(image).to(self.device).unsqueeze(0)
        mask_gt = mask_gt.squeeze()

        with torch.no_grad():
            mask_pred = self.model(image).cpu().numpy().squeeze().round()

        overlay_image = get_overlay_image(image_vis, mask_gt, mask_pred)

        self.tb_logger.writer.add_image(
            "example prediction", overlay_image, epoch, dataformats="HWC"
        )
