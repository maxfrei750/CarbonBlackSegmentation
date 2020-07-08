import ignite.distributed as idist
import segmentation_models_pytorch as smp
from ignite.contrib.handlers import PiecewiseLinear
from torch import optim


def get_optimizer(model, config):

    assert config["optimizer"] in optim.__dict__, f"Unknown optimizer: {config['optimizer']}"

    optimizer = optim.__dict__[config["optimizer"]](model.parameters(), lr=config["learning_rate"])

    optimizer = idist.auto_optim(optimizer)

    return optimizer


def get_loss(config):
    return smp.utils.losses.__dict__[config["loss"]]().to(idist.device())


def get_lr_scheduler(optimizer, config):
    le = config["num_iters_per_epoch"]
    milestones_values = [
        (0, 0.0),
        (le * config["num_warmup_epochs"], config["learning_rate"]),
        (le * config["num_epochs"], 0.0),
    ]
    lr_scheduler = PiecewiseLinear(optimizer, param_name="lr", milestones_values=milestones_values)
    return lr_scheduler
