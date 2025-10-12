import random
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF
from PIL import Image


def get_mean_and_std(dataset, batch_size=64):
    """
        Calculates the mean and standard deviation for dataset
        within a given batch

        Args:
            dataset: the dataset used to train model
            batch_size: batch size for dataloader
    """

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    n_images = 0 # keeps running total of number of images
    mean = 0.0
    std = 0.0

    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1) # reshapes so we can compute the statistics all at once instead of row by row

        mean += images.mean(2).sum(0) # sums up the mean per channel
        std += images.std(2).sum(0) # sums up the std per channel
        n_images += batch_samples

    mean /= n_images
    std /= n_images
    return mean, std

def mask_to_tensor(mask):
    return torch.as_tensor(np.array(mask), dtype=torch.long) - 1

def get_transforms(size):
    """"
        Applies the necessary transforms onto image and mask pairs

        Args:
            mean: mean of dataset used to normalize input image
            std: standard deviation of dataset used to normalize input image
    """
    image_transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
        ])
    mask_transform = transforms.Compose([
        transforms.Resize((size,size), InterpolationMode.NEAREST), # preserve tensor values for mask (1,2,3)
        mask_to_tensor
        ])
    return (image_transform, mask_transform)

def get_simple_transforms():
    """
        Used so that get_mean_std works on a properly transformed dataset
    """
    simple_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
    ])
    return simple_transform

def relu(x):
    """ ReLU activation """
    return max(0, x)