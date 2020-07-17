import numpy as np

import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.OneOf([albu.CLAHE(p=1), albu.RandomBrightness(p=1), albu.RandomGamma(p=1)], p=0.9),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf([albu.RandomContrast(p=1), albu.HueSaturationValue(p=1)], p=0.9),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.PadIfNeeded(384, 480)]
    return albu.Compose(test_transform)


def to_tensor_image(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def to_tensor_mask(x, **kwargs):
    return np.expand_dims(x, axis=0).astype("float32")


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor_image, mask=to_tensor_mask),
    ]
    return albu.Compose(_transform)


def test(data_root):
    from PIL import Image
    from data import SegmentationDataset
    from visualization import get_overlay_image

    dataset = SegmentationDataset(data_root, "val", augmentation=get_training_augmentation())

    image, mask, _ = dataset.get_example_sample()

    overlay_image = get_overlay_image(image, mask)
    Image.fromarray(overlay_image).show()


if __name__ == "__main__":
    import fire

    fire.Fire(test)
