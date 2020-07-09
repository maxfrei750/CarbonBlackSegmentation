def setup_trains_logging(config):
    if config["with_trains"]:
        from trains import Task

        task = Task.init("Carbon Black Semantic Segmentation Training", config["task_name"])
        task.connect_configuration(config)

        # Log hyper parameters
        hyper_parameters = list(config.keys())
        task.connect({k: config[k] for k in hyper_parameters})
