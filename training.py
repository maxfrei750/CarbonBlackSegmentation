from pathlib import Path

import modeling
import segmentation_models_pytorch as smp
import torch
import utils
from data import get_dataloaders
from ignite import distributed as idist
from ignite.contrib.engines import common
from ignite.contrib.handlers import create_lr_scheduler_with_warmup
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import Checkpoint
from ignite.utils import manual_seed, setup_logger
from log import (
    ExamplePredictionLogger,
    get_save_handler,
    log_basic_info,
    log_confusion_matrix,
    log_metrics,
    setup_trains_logging,
)
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from validation import get_metrics


def get_optimizer(model, config):

    assert config["optimizer"] in optim.__dict__, f"Unknown optimizer: {config['optimizer']}"

    optimizer = optim.__dict__[config["optimizer"]](
        model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
    )

    optimizer = idist.auto_optim(optimizer)

    return optimizer


def get_loss():
    return smp.utils.losses.BCELoss().to(idist.device())


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


def create_trainer(model, optimizer, criterion, lr_scheduler, train_sampler, config, logger):

    device = idist.device()

    # Setup Ignite trainer:
    # - let's define training step
    # - add other common handlers:
    #    - TerminateOnNan,
    #    - handler to setup learning rate scheduling,
    #    - ModelCheckpoint
    #    - RunningAverage` on `train_step` output
    #    - Two progress bars on epochs and optionally on iterations

    def train_step(engine, batch):

        x, y = batch[0], batch[1]

        if x.device != device:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

        model.train()
        # Supervised part
        y_pred = model(x)

        criterion.weights = torch.tensor(
            [1 - torch.sum(y == x).to(torch.float) / torch.numel(y) for x in [0, 1]], device=device
        )

        loss = criterion(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # This can be helpful for XLA to avoid performance slow down if fetch loss.item() every iteration
        if (
            config["log_every_iters"] > 0
            and (engine.state.iteration - 1) % config["log_every_iters"] == 0
        ):
            batch_loss = loss.item()
            engine.state.saved_batch_loss = batch_loss
        else:
            batch_loss = engine.state.saved_batch_loss

        return {"batch loss": batch_loss}

    trainer = Engine(train_step)
    trainer.state.saved_batch_loss = -1.0
    trainer.state_dict_user_keys.append("saved_batch_loss")
    trainer.logger = logger

    to_save = {
        "trainer": trainer,
        "model": model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }
    metric_names = ["batch loss"]

    common.setup_common_training_handlers(
        trainer=trainer,
        train_sampler=train_sampler,
        to_save=to_save,
        save_every_iters=config["checkpoint_every"],
        save_handler=get_save_handler(config),
        lr_scheduler=lr_scheduler,
        output_names=metric_names if config["log_every_iters"] > 0 else None,
        with_pbars=False,
        clear_cuda_cache=False,
    )

    resume_from = config["resume_from"]
    if resume_from is not None:
        checkpoint_fp = Path(resume_from)
        assert checkpoint_fp.exists(), f"Checkpoint '{checkpoint_fp.as_posix()}' is not found"
        logger.info(f"Resume from a checkpoint: {checkpoint_fp.as_posix()}")
        checkpoint = torch.load(checkpoint_fp.as_posix(), map_location="cpu")
        Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

    return trainer


def training(local_rank, config):

    rank = idist.get_rank()
    manual_seed(config["seed"] + rank)

    device = idist.device()

    logger = setup_logger(
        name="Carbon Black Semantic Segmentation Training", distributed_rank=local_rank
    )

    log_basic_info(logger, config)

    output_path = config["output_path"]

    if rank == 0:
        if config["stop_iteration"] is None:
            now = utils.get_time_stamp()
        else:
            now = f"stop-on-{config['stop_iteration']}"

        folder_name = (
            f"{config['architecture']}-{config['encoder']}-{config['encoder_weights']}_"
            f"backend-{idist.backend()}-{idist.get_world_size()}_{now}"
        )

        output_path = Path(output_path) / folder_name
        output_path.mkdir(parents=True, exist_ok=True)
        config["output_path"] = output_path.as_posix()
        config["task_name"] = output_path.stem

        logger.info(f"Output path: {output_path}")

        if "cuda" in idist.device().type:
            config["cuda_device_name"] = torch.cuda.get_device_name(local_rank)

        setup_trains_logging(config)

    dataloader_train, dataloader_val = get_dataloaders(config)

    config["num_iterations_per_epoch"] = len(dataloader_train)
    config["num_epochs"] = round(config["num_iterations"] / config["num_iterations_per_epoch"])
    model = modeling.get_model(config)

    optimizer = get_optimizer(model, config)
    loss = get_loss()

    lr_scheduler = get_lr_scheduler(optimizer, config)

    trainer = create_trainer(
        model, optimizer, loss, lr_scheduler, dataloader_train.sampler, config, logger
    )

    metrics = get_metrics(loss)

    # We define two evaluators as they wont have exactly similar roles:
    # - `evaluator` will save the best model based on validation score
    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device, non_blocking=True
    )
    evaluator_train = create_supervised_evaluator(
        model, metrics=metrics, device=device, non_blocking=True
    )

    if rank == 0:
        # Setup TensorBoard logging on trainer and evaluators. Logged values are:
        #  - Training metrics, e.g. running average loss values
        #  - Learning rate
        #  - Evaluation train/test metrics
        evaluators = {"training": evaluator_train, "validation": evaluator}
        tb_logger = common.setup_tb_logging(output_path, trainer, optimizer, evaluators=evaluators)

        example_prediction_logger = ExamplePredictionLogger(tb_logger, model, device)

    def run_validation(engine):
        epoch = trainer.state.epoch
        state = evaluator_train.run(dataloader_train)
        data_subset = "Train"
        log_metrics(logger, epoch, state.times["COMPLETED"], data_subset, state.metrics)
        log_confusion_matrix(tb_logger, epoch, data_subset, state.metrics)

        state = evaluator.run(dataloader_val)
        data_subset = "Val"
        log_metrics(logger, epoch, state.times["COMPLETED"], data_subset, state.metrics)
        log_confusion_matrix(tb_logger, epoch, data_subset, state.metrics)
        example_prediction_logger.log_visualization(dataloader_val.dataset, epoch)

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED(every=config["validate_every"]) | Events.COMPLETED, run_validation
    )

    # Store 3 best models by validation accuracy:
    common.gen_save_best_models_by_val_score(
        save_handler=get_save_handler(config),
        evaluator=evaluator,
        models={"model": model},
        metric_name="accuracy",
        n_saved=3,
        trainer=trainer,
        tag="validation",
    )

    # TODO: Add early stopping

    # In order to check training resuming we can stop training on a given iteration
    if config["stop_iteration"] is not None:

        @trainer.on(Events.ITERATION_STARTED(once=config["stop_iteration"]))
        def _():
            logger.info(f"Stop training on {trainer.state.iteration} iteration")
            trainer.terminate()

    # noinspection PyBroadException
    try:
        trainer.run(dataloader_train, max_epochs=config["num_epochs"])
    except Exception:
        import traceback

        print(traceback.format_exc())

    if rank == 0:
        # noinspection PyUnboundLocalVariable
        tb_logger.close()
