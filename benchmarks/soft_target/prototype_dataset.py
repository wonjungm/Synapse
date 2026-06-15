import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor

CIFAR_MEAN_STD = {
    "mean": [0.5071, 0.4865, 0.4409],
    "std": [0.2673, 0.2564, 0.2761]
}

IMAGENET_MEAN_STD = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
}

def get_transforms(image_size=224, dataset_name="imagenet"):
    if dataset_name.startswith("cifar"):
        stats = CIFAR_MEAN_STD
    else:
        stats = IMAGENET_MEAN_STD

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=stats["mean"], std=stats["std"]),
    ])

def get_dataset(dataset_name, root, transform):
    if dataset_name == "cifar10":
        return datasets.CIFAR10(root=root, train=True, download=True, transform=transform), \
               datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    elif dataset_name == "cifar100":
        return datasets.CIFAR100(root=root, train=True, download=True, transform=transform), \
               datasets.CIFAR100(root=root, train=False, download=True, transform=transform)

    elif dataset_name == "imagenet2012":
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join(root, "val")
        return datasets.ImageFolder(train_dir, transform=transform), \
               datasets.ImageFolder(val_dir, transform=transform)

    elif dataset_name == "imagenet_subset_100k":
        train_dir = os.path.join(root, "train")
        val_dir = os.path.join('/nas-ssd/datasets/imagenet2012/imagenet', "val") if os.path.exists(os.path.join('/nas-ssd/datasets/imagenet2012/imagenet', "val")) else None
        train_set = datasets.ImageFolder(train_dir, transform=transform)
        val_set = datasets.ImageFolder(val_dir, transform=transform) if val_dir else None
        return train_set, val_set

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_dataloaders(dataset_name,
                    root,
                    batch_size=64,
                    num_workers=4,
                    image_size=224,
                    shuffle_train=True,
                    pin_memory=True):

    transform = get_transforms(image_size=image_size, dataset_name=dataset_name)
    train_set, val_set = get_dataset(dataset_name, root=root, transform=transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=shuffle_train,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    val_loader = None
    if val_set is not None:
        val_loader = DataLoader(val_set,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=pin_memory)

    return train_loader, val_loader

