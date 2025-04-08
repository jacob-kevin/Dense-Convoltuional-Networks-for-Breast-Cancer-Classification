# data_loader.py

import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, ConcatDataset

def get_dataloaders(data_dir, batch_size=32):
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    train_dataset = ImageFolder(root=train_dir, transform=data_transforms)
    val_dataset = ImageFolder(root=val_dir, transform=data_transforms)
    test_dataset = ImageFolder(root=test_dir, transform=test_transforms)

    combined_dataset = ConcatDataset([train_dataset, val_dataset])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return combined_dataset, test_loader

