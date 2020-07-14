from pathlib import Path

import fire
import ignite
import ignite.distributed as idist
import modeling
import torch
import utils
from data import get_dataloaders
from evaluation import ExamplePredictionLogger, get_metrics
from ignite.contrib.engines import common
from ignite.engine import Engine, Events, create_supervised_evaluator
from ignite.handlers import Checkpoint, DiskSaver
from ignite.utils import manual_seed, setup_logger
from log import setup_trains_logging
from training import get_loss, get_lr_scheduler, get_optimizer
from visualization import plot_confusion_matrix

# TODO: Refactoring
# TODO: Freeze resnet stages.


def log_confusion_matrix(tb_logger, epoch, data_subset, metrics):
    figure = plot_confusion_matrix(metrics["confusion matrix"].cpu().numpy())
    tb_logger.writer.add_figure(f"confusion matrix/{data_subset}", figure, global_step=epoch)


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
    loss = get_loss(config)

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


def run(
    seed=42,
    data_path="./data",
    subset_train="train",
    subset_val="val",
    output_path="./output",
    architecture="FPN",
    encoder="resnet50",
    encoder_weights="imagenet",
    batch_size=6,
    optimizer="Adam",
    weight_decay=1e-4,
    loss="BCELoss",
    num_workers=12,
    num_iterations=10000,
    learning_rate=0.0001,
    learning_rate_milestone_iterations=(2000, 8000),
    gamma=0.1,
    num_warmup_iterations=1000,
    warmup_factor=0.001,
    validate_every=10,
    checkpoint_every=200,
    backend=None,
    resume_from=None,
    log_every_iters=0,
    nproc_per_node=None,
    stop_iteration=None,
    with_trains=False,
    **spawn_kwargs,
):
    """Main entry to train a model on the semantic segmentation of carbon black agglomerate TEM images.

    Args:
        seed (int): random state seed to set. Default, 42.
        data_path (str): input dataset path. Default, "/tmp/cifar10".
        subset_train (str): name of training subset. Default, "train".
        subset_val (str): name of validation subset. Default, "val".
        architecture (str): architecture (see https://github.com/qubvel/segmentation_models.pytorch#architectures-).
            Default, "FPN".
        encoder (str): encoder architecture (see https://github.com/qubvel/segmentation_models.pytorch#encoders-).
            Default, "resnet50".
        encoder_weights (str): pretrained weights (see https://github.com/qubvel/segmentation_models.pytorch#encoders-).
            Default, "imagenet".
        output_path (str): output path. Default, "./data".
        batch_size (int): total batch size. Default, 8.
        optimizer (str): optimizer. Default, "Adam".
        weight_decay (float): weight decay. Default, 1e-4.
        loss (string): loss. Default, "DiceLoss".
        num_workers (int): number of workers in the data loader. Default, 12.
        num_iterations (int): number of iterations to train the model. Default, 10000.
        learning_rate (float): peak of piecewise linear learning rate scheduler. Default, 0.4.
        learning_rate_milestone_iterations (iterable of int): numbers of iterations where learning rate is each time
            decreased by a factor gamma. Default, (2000, 8000).
        gamma (float): factor to multiply learning rate with at each milestone. Default, 0.1.
        num_warmup_iterations (int): number of warm-up iterations before learning rate decay. Default, 1000.
        warmup_factor (float): learning rate starts at warmup_factor * learning_rate. Default, 0.001.
        validate_every (int): run model's validation every ``validate_every`` epochs. Default, 3.
        checkpoint_every (int): store training checkpoint every ``checkpoint_every`` iterations. Default, 200.
        backend (str, optional): backend to use for distributed configuration. Possible values: None, "nccl", "xla-tpu",
            "gloo" etc. Default, None.
        nproc_per_node (int, optional): optional argument to setup number of processes per node. It is useful,
            when main python process is spawning training as child processes.
        resume_from (str, optional): path to checkpoint to use to resume the training from. Default, None.
        log_every_iters (int): argument to log batch loss every ``log_every_iters`` iterations.
            It can be 0 to disable it. Default, 15.
        stop_iteration (int, optional): iteration to stop the training. Can be used to check resume from checkpoint.
        with_trains (bool): if True, experiment Trains logger is setup. Default, False.
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes

    """
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    spawn_kwargs["nproc_per_node"] = nproc_per_node

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:

        parallel.run(training, config)


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


def get_save_handler(config):
    if config["with_trains"]:
        from ignite.contrib.handlers.trains_logger import TrainsSaver

        return TrainsSaver(dirname=config["output_path"])

    return DiskSaver(config["output_path"], require_empty=False)


if __name__ == "__main__":
    active_gpu_ids = [0, 1, 2, 3]
    utils.select_active_gpus(active_gpu_ids)

    fire.Fire(run)
