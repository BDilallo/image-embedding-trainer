from typing import Callable, Tuple

from PIL import Image, ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import TrainConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True

_CHANNELS_TO_PIL_MODE = {1: "L", 3: "RGB"}


def _build_loader(num_channels: int) -> Callable[[str], Image.Image]:
    """torchvision's default ImageFolder loader always converts to RGB,
    which silently discards num_channels for grayscale datasets. This
    loader instead converts to whatever PIL mode matches num_channels."""

    mode = _CHANNELS_TO_PIL_MODE[num_channels]

    def loader(path: str) -> Image.Image:
        with open(path, "rb") as file:
            image = Image.open(file)
            return image.convert(mode)

    return loader


def build_transforms(config: TrainConfig) -> Tuple[transforms.Compose, transforms.Compose]:
    image_size = config.image_size
    resize_size = int(image_size * 256 / 224)

    train_transform_steps = [
        transforms.Resize(resize_size),
        transforms.RandomResizedCrop(image_size, scale=config.random_crop_scale),
    ]

    if config.use_horizontal_flip:
        train_transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))

    if config.color_jitter is not None:
        brightness, contrast, saturation = config.color_jitter
        train_transform_steps.append(
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            )
        )

    train_transform_steps += [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std,
        ),
    ]

    train_transforms = transforms.Compose(train_transform_steps)

    val_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config.normalize_mean,
            std=config.normalize_std,
        ),
    ])

    return train_transforms, val_transforms


def build_dataloaders(
    config: TrainConfig,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder, DataLoader, DataLoader]:
    """
    Folder structure (subfolder name = class label):

    train_dir/
        class_1/
            image1.png
            image2.png
        class_2/
            image1.png

    val_dir/
        class_1/
            image1.png
        class_2/
            image1.png

    Any domain that can be organized this way works - the folder names
    are not tied to any particular subject (faces, characters, products,
    etc.).
    """

    train_transforms, val_transforms = build_transforms(config)
    loader = _build_loader(config.num_channels)

    train_dataset = datasets.ImageFolder(
        root=config.train_dir,
        transform=train_transforms,
        loader=loader,
    )

    val_dataset = datasets.ImageFolder(
        root=config.val_dir,
        transform=val_transforms,
        loader=loader,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )

    return train_dataset, val_dataset, train_loader, val_loader
