import argparse
import os

import segmentation_models_pytorch as smp
import torch
from data import SegmentationDataset
from torch.utils.data import DataLoader
from transforms import get_preprocessing, get_training_augmentation, get_validation_augmentation
from utils import get_time_stamp


def get_model(architecture, encoder_name, encoder_weights):
    """For a list of possible architectures, encoders and weights, please refer to:
        https://github.com/qubvel/segmentation_models.pytorch#architectures-
    """
    assert architecture in smp.__dict__, f"Unknown architecture: {architecture}"

    return smp.__dict__[architecture](
        encoder_name=encoder_name, encoder_weights=encoder_weights, classes=1, activation="sigmoid"
    )


def main(args):

    model = get_model(args.architecture, args.encoder, args.encoder_weights)
    preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder, args.encoder_weights)

    dataset_train = SegmentationDataset(
        args.data_root,
        "train",
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    dataset_val = SegmentationDataset(
        args.data_root,
        "test",
        augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(preprocessing_fn),
    )

    dataloader_train = DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=12
    )
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4)

    # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index
    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5), smp.utils.metrics.Accuracy()]

    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.learning_rate)])

    train_epoch = smp.utils.train.TrainEpoch(
        model, loss=loss, metrics=metrics, optimizer=optimizer, device=args.device, verbose=True
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model, loss=loss, metrics=metrics, device=args.device, verbose=True
    )

    output_path = os.path.join(args.output_root, get_time_stamp())
    os.makedirs(output_path, exist_ok=True)

    max_score = 0

    for i in range(0, args.epochs):

        print("\nEpoch: {}".format(i))
        logs_train = train_epoch.run(dataloader_train)
        logs_val = valid_epoch.run(dataloader_val)

        if max_score < logs_val["iou_score"]:
            max_score = logs_val["iou_score"]
            torch.save(model, os.path.join(output_path, "best_model.pth"))
            print("Model saved!")

        if i == 25:
            optimizer.param_groups[0]["lr"] = 1e-5
            print("Decrease decoder learning rate to 1e-5!")


def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Segmentation Training")

    parser.add_argument("data_root", help="data root")
    parser.add_argument(
        "--subset-test", default="test", help="name of the data subset used for testing the model"
    )
    parser.add_argument(
        "--subset-train",
        default="train",
        help="name of the data subset used for training the model",
    )

    parser.add_argument(
        "--architecture",
        default="FPN",
        help="architecture (see https://github.com/qubvel/segmentation_models.pytorch#architectures-)",
    )

    parser.add_argument(
        "--encoder",
        default="resnet50",
        help="encoder (see https://github.com/qubvel/segmentation_models.pytorch#encoders-)",
    )

    parser.add_argument(
        "--encoder-weights",
        default="imagenet",
        help="encoder weights (see https://github.com/qubvel/segmentation_models.pytorch#encoders-)",
    )

    parser.add_argument("--device", default="cuda", help="device")
    parser.add_argument("-b", "--batch-size", default=8, type=int)
    parser.add_argument(
        "-e", "--epochs", default=40, type=int, metavar="N", help="number of total epochs to run"
    )

    parser.add_argument("--learning-rate", default=0.0001, type=float, help="initial learning rate")
    # parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    # parser.add_argument(
    #     "--wd",
    #     "--weight-decay",
    #     default=1e-4,
    #     type=float,
    #     metavar="W",
    #     help="weight decay (default: 1e-4)",
    #     dest="weight_decay",
    # )
    parser.add_argument("-o", "--output-root", default="output", help="path where to save")
    parser.add_argument("--resume", default="", help="resume from checkpoint")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
