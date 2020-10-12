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

    if "encoder_freeze_at" in config:
        freeze_encoder_at(model.encoder, config["encoder_freeze_at"])

    if "device" in config:
        if config["device"] == "cpu":
            return model

    # Adapt model for distributed settings if configured
    model = idist.auto_model(model)

    return model


def freeze_encoder_at(encoder, freeze_at):
    if freeze_at is not None:
        layer_id = freeze_at
        while hasattr(encoder, f"layer{layer_id}"):
            for parameter in getattr(encoder, f"layer{layer_id}").parameters():
                parameter.requires_grad = False

            layer_id += 1
