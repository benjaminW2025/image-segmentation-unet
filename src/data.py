import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import os
import glob
import random

class OxfordPetsSegmentation(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None, seed=0):
        """
        Initialize the custom dataset object

        Args:
            root: file where the images and masks are stored
            split: the split of data desired, either training, test, or validation
            transform: the transforms applied onto the image
            target_transform: the transforms applied onto the masks
            seed: random seed used to ensure that training splits are distributed correctly
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed

        self.images_dir = Path(root) / "images" # creates a path to "images" file
        self.masks_dir = Path(root) / "annotations" / "trimaps" # creates a path to "trimaps" file
        
        images_list = sorted(self.images_dir.glob("*.jpg"))
        masks_list = sorted(self.masks_dir.glob("*.png"))
        self.samples = list(zip(images_list, masks_list)) # builds a list of tuples of images and masks

        random.seed(self.seed)
        random.shuffle(self.samples) # allows us to create an OxfordPetSegmentation object with the same ordering for each split
        n = len(self.samples)
        if (self.split == "train"):
            self.samples = self.samples[:int(0.8 * n)]
        elif (self.split == "val"):
            self.samples = self.samples[int(0.8 * n):int(0.9 * n)]
        elif (self.split == "test"):
            self.samples = self.samples[int(0.9 * n):]



    def __len__(self):
        """return number of samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """Returns (image, mask) tuple with preprocessing applied"""
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path)
        mask = Image.open(mask_path)

        # ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)  # applies the images transforms
        if self.target_transform:
            mask = self.target_transform(mask) # applies the masks transforms
        return (image, mask) # returns the transformed pair as a tuple