import random
from src.train import train, val_test, check_accuracy
from src.utils import get_mean_and_std, get_transforms, get_simple_transforms, relu
from src.model import UNet
from src.data import OxfordPetsSegmentation
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def main():
    tensor_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()])

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Using device:", device)

    model = UNet(initial_channels=3, num_classes=3, features=64, batchnorm=True)
    model.to(device)

    num = random.randint(0, 1000) # generates the seed

    # define transforms
    img_transform, mask_transform = get_transforms(128)

    # define loss function
    criterion = nn.CrossEntropyLoss()

    # clear old gradients
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # create the dataloaders used for train / val / test

    dataset_train = OxfordPetsSegmentation(root="data/oxford_pets", split="train", transform=img_transform, target_transform=mask_transform, seed = num)
    dataset_val = OxfordPetsSegmentation(root="data/oxford_pets", split="val", transform=img_transform, target_transform=mask_transform, seed = num)
    dataset_test = OxfordPetsSegmentation(root="data/oxford_pets", split="test", transform=img_transform, target_transform=mask_transform, seed = num)
    train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(dataset_val, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False, num_workers=0)

    # training loop
    num_epochs = 5

    train(5, model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = val_test(model, val_loader, criterion, device)

    print("Pixel-wise Accuracy on Validation Set:")
    check_accuracy(model, val_loader, device)

if __name__ == "__main__":
    main()
