import fire
import ignite.distributed as idist
import utils
from training import training

# TODO: Use larger image chunks.


def run(
    seed=42,
    data_path="./data",
    subset_train="train",
    subset_val="val",
    output_path="./output",
    architecture="FPN",
    encoder="resnet50",
    encoder_weights="imagenet",
    encoder_freeze_at=None,
    batch_size=6,
    optimizer="Adam",
    weight_decay=1e-4,
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
    active_gpu_ids=(0,),
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
        encoder_freeze_at (int or None): defines stages of the encoder which are frozen before the training (e.g. 2
            means all stages including stage 2 and beyond). Default, None.
        output_path (str): output path. Default, "./data".
        batch_size (int): total batch size. Default, 8.
        optimizer (str): optimizer. Default, "Adam".
        weight_decay (float): weight decay. Default, 1e-4.
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
        active_gpu_ids (tuple of int): ids of GPUs to use. Default, (0,).
        **spawn_kwargs: Other kwargs to spawn run in child processes: master_addr, master_port, node_rank, nnodes

    """
    # catch all local parameters
    config = locals()
    config.update(config["spawn_kwargs"])
    del config["spawn_kwargs"]

    utils.select_active_gpus(config["active_gpu_ids"])

    spawn_kwargs["nproc_per_node"] = nproc_per_node

    with idist.Parallel(backend=backend, **spawn_kwargs) as parallel:
        parallel.run(training, config)


if __name__ == "__main__":
    fire.Fire(run)
