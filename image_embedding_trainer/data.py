from typing import Tuple
from PIL import ImageFile
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import TrainConfig

ImageFile.LOAD_TRUNCATED_IMAGES = True

def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    resize_size = int(image_size * 256 / 224)

    train_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.10,
            contrast=0.10,
            saturation=0.08,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])

    return train_transforms, val_transforms


def build_dataloaders(
    config: TrainConfig,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder, DataLoader, DataLoader]:
    """
    Folder structure:

    train_dir/
        character_1/
            image1.png
            image2.png
        character_2/
            image1.png

    val_dir/
        character_1/
            image1.png
        character_2/
            image1.png

    subfolder = class label.
    """

    train_transforms, val_transforms = build_transforms(config.image_size)

    train_dataset = datasets.ImageFolder(
        root=config.train_dir,
        transform=train_transforms,
    )

    val_dataset = datasets.ImageFolder(
        root=config.val_dir,
        transform=val_transforms,
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