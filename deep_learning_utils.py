import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Function to de-normalize images
def denormalize_image(tensor, mean, std):
    tensor = tensor.clone()  # Create a copy to avoid modifying the original
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse the normalization operation
    return tensor

# Function to visualize a batch of images
def show_images_from_loader(loader, parent_dataset, mean, std):
    """
    Display a batch of preprocessed images with their labels.

    Args:
        loader (DataLoader): The DataLoader from which to extract a batch.
        parent_dataset (Dataset): The parent dataset (e.g., train_dataset_full).
        mean (list): Mean used for normalization.
        std (list): Standard deviation used for normalization.
    """
    classes = parent_dataset.classes  # Get the class names from the parent dataset
    data_iter = iter(loader)
    images, labels = next(data_iter)  # Retrieve a batch

    # De-normalize the batch
    images = images.clone()
    for i in range(images.size(0)):
        images[i] = denormalize_image(images[i], mean, std)

    # Convert to numpy format for matplotlib
    images = images.numpy().transpose((0, 2, 3, 1))  # From CxHxW to HxWxC

    # Display the first 4 images in the batch
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i, ax in enumerate(axes):
        if i >= len(images):  # Avoid errors if there are fewer than 4 images in the batch
            break
        ax.imshow(np.clip(images[i], 0, 1))  # Ensure values are in the range [0, 1]
        ax.axis("off")
        ax.set_title(f"Label: {classes[labels[i]]}")
    plt.show()

def create_dataloaders(dataset_path, train_split=0.7, val_split=0.2, test_split=0.1, batch_size=32):

    # Transform images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Compute sets size
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Create DataLoader for each part of the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Function to de-normalize images
def denormalize_image(tensor, mean, std):
    tensor = tensor.clone()  # Create a copy to avoid modifying the original
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Reverse the normalization operation
    return tensor

# Function to visualize a batch of images
def show_images_from_loader(loader, parent_dataset, mean, std):
    """
    Display a batch of preprocessed images with their labels.

    Args:
        loader (DataLoader): The DataLoader from which to extract a batch.
        parent_dataset (Dataset): The parent dataset (e.g., train_dataset_full).
        mean (list): Mean used for normalization.
        std (list): Standard deviation used for normalization.
    """
    classes = parent_dataset.classes  # Get the class names from the parent dataset
    data_iter = iter(loader)
    images, labels = next(data_iter)  # Retrieve a batch

    # De-normalize the batch
    images = images.clone()
    for i in range(images.size(0)):
        images[i] = denormalize_image(images[i], mean, std)

    # Convert to numpy format for matplotlib
    images = images.numpy().transpose((0, 2, 3, 1))  # From CxHxW to HxWxC

    # Display the first 4 images in the batch
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))
    for i, ax in enumerate(axes):
        if i >= len(images):  # Avoid errors if there are fewer than 4 images in the batch
            break
        ax.imshow(np.clip(images[i], 0, 1))  # Ensure values are in the range [0, 1]
        ax.axis("off")
        ax.set_title(f"Label: {classes[labels[i]]}")
    plt.show()

def create_dataloaders(dataset_path, train_split=0.7, val_split=0.2, test_split=0.1, batch_size=32):

    # Transform images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    full_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Compute sets size
    dataset_size = len(full_dataset)
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Split the dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # Create DataLoader for each part of the data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%")

        # Validation loop
        model.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_accuracy = 100 * correct_val / total_val
        print(f"Validation Acc: {val_accuracy:.2f}%")
