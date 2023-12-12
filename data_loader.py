import os
import torch
from PIL import Image, ImageFile
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter
from config import Config

# Ignore corrupt EXIF data
ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_dataset():
    """Load and transform the dataset with balanced sampling for classes."""
    # Custom loader to handle images with corrupt EXIF data
    def pil_loader(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = datasets.ImageFolder(root=Config.DATASET_DIRECTORY, transform=transform, loader=pil_loader)

    # Display dataset folder to class label mapping
    print("Dataset folder to class label mapping:")
    for class_label, class_name in enumerate(full_dataset.classes):
        print(f"Dataset '{class_name}' as Class {class_label}")

    # Calculate class weights
    class_counts = Counter([label for _, label in full_dataset.samples])
    class_weights = [1.0 / class_counts[label] for _, label in full_dataset.samples]
    sample_weights = torch.DoubleTensor(class_weights)

    # Split dataset into train, validation, and test
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - (train_size + val_size)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size, test_size])

    # Correctly apply weighted sampler for the train dataset
    train_indices = train_dataset.indices
    train_weights = [sample_weights[i] for i in train_indices]
    weighted_sampler = WeightedRandomSampler(train_weights, len(train_weights))

    # Data loaders with weighted sampling for the training set
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=weighted_sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=4)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_dataset()
