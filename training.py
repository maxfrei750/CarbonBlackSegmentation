import ignite.distributed as idist
import segmentation_models_pytorch as smp
from ignite.contrib.handlers import create_lr_scheduler_with_warmup
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR


def get_optimizer(model, config):

    assert config["optimizer"] in optim.__dict__, f"Unknown optimizer: {config['optimizer']}"

    optimizer = optim.__dict__[config["optimizer"]](
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    optimizer = idist.auto_optim(optimizer)

    return optimizer


def get_loss(config):
    return smp.utils.losses.__dict__[config["loss"]]().to(idist.device())


def get_lr_scheduler(optimizer, config):
    lr = config["learning_rate"]
    warmup_factor = config["warmup_factor"]
    num_warmup_iterations = config["num_warmup_iterations"]
    learning_rate_milestone_iterations = config["learning_rate_milestone_iterations"]
    gamma = config["gamma"]

    learning_rate_milestone_iterations = [
        x - num_warmup_iterations for x in learning_rate_milestone_iterations
    ]
    lr_scheduler = MultiStepLR(
        optimizer=optimizer, gamma=gamma, milestones=learning_rate_milestone_iterations
    )

    lr_scheduler = create_lr_scheduler_with_warmup(
        lr_scheduler,
        warmup_start_value=lr * warmup_factor,
        warmup_end_value=lr,
        warmup_duration=num_warmup_iterations,
    )
    return lr_scheduler
