def setup_trains_logging(config):
    if config["with_trains"]:
        from trains import Task

        task = Task.init("Carbon Black Semantic Segmentation Training", config["task_name"])
        task.connect_configuration(config)

        # Log hyper parameters
        hyper_params = [
            "seed",
            "data_path",
            "subset_train",
            "subset_val",
            "architecture",
            "encoder",
            "encoder_weights",
            "batch_size",
            "optimizer",
            "momentum",
            "weight_decay",
            "num_epochs",
            "learning_rate",
            "num_warmup_epochs",
            "loss",
        ]
        task.connect({k: config[k] for k in hyper_params})
