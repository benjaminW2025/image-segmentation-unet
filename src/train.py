import random
from src.data import OxfordPetsSegmentation
from src.model import UNet
from src.utils import get_mean_and_std, get_transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

def train(num_epochs, model, train_loader, criterion, optimizer, device):
    """
        Implements the training loop for the model which handles loss computation, backpropogation,  and weight optimization

        Args:
            num_epochs = number of epochs that the batch is trained on
    """ 
    for epochs in range(num_epochs):
        model.train() # puts Pytorch in training mode   

        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            # forward pass, backpropogation, loss computation, update weights
            outputs = model(images)
            loss = criterion(outputs, masks)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def val_test(model, loader, criterion, device):
    """
        Runs the val/test loops
    """
    model.eval()
    val_loss = 0.0
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            masks = masks.long()
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            # compute pixel-wise accuracy (simple metric)
            preds = torch.argmax(outputs, dim=1)  # shape: (B, H, W)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += torch.numel(masks)

    avg_loss = val_loss / len(loader)
    pixel_acc = correct_pixels / total_pixels
    return avg_loss, pixel_acc

def check_accuracy(model, dataloader, device):
    """
    Computes pixel-wise accuracy on a dataset.
    
    Args:
        model: trained PyTorch model
        dataloader: DataLoader for validation/test set
        device: "cpu", "cuda", or "mps"
    
    Returns:
        pixel_accuracy: float
    """
    model.eval()  # set model to evaluation mode
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)  # forward pass
            preds = torch.argmax(outputs, dim=1)  # get predicted class per pixel

            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()  # total number of pixels

    pixel_accuracy = correct_pixels / total_pixels
    print(f"Pixel-wise Accuracy: {pixel_accuracy*100:.2f}%")
    return pixel_accuracy