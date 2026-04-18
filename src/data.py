import torch
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler

def get_dataloader(data_dir, batch_size=64):

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    # =========================
    # CLASS BALANCING (IMPORTANT)
    # =========================
    targets = [label for _, label in dataset]

    class_count = np.bincount(targets)
    class_weights = 1. / class_count

    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler   # instead of shuffle
    )

    return loader