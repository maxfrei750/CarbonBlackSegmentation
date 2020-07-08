import ignite.distributed as idist
import segmentation_models_pytorch as smp


def get_model(config):
    """For a list of possible architectures, encoders and weights, please refer to:
        https://github.com/qubvel/segmentation_models.pytorch#architectures-
    """
    assert config["architecture"] in smp.__dict__, f"Unknown architecture: {config['architecture']}"

    model = smp.__dict__[config["architecture"]](
        encoder_name=config["encoder"],
        encoder_weights=config["encoder_weights"],
        classes=1,
        activation="sigmoid",
    )

    # Adapt model for distributed settings if configured
    model = idist.auto_model(model)

    return model
