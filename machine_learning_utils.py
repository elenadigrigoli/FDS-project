import os
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader




# Function to load images from a directory
def load_images(root_dir, image_size=(256, 256), gray=False):
    images = []
    labels = []
    class_names = []    
    
    # Traverse through class folders
    for class_index, class_name in enumerate(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            class_names.append(class_name)
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                # Read and resize image
                if gray:
                    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
                else:
                    img = cv.imread(image_path, cv.IMREAD_COLOR)
                if img is not None:  # Ensure image is valid
                    img = cv.resize(img, image_size, interpolation=cv.INTER_AREA)
                    images.append(img)
                    labels.append(class_index)  # Label as the folder index
    return np.array(images), np.array(labels), class_names



def calculate_histograms(image, norm=False):
    histograms = []
    if norm:
        bins = 256
        for channel in range(image.shape[2]):  # Iterate on B, G, R
            hist = cv.calcHist([image], [channel], None, [bins], [0, 1])
            histograms.append(hist.flatten())
    else:
        for channel in range(image.shape[2]):  # Iterate on B, G, R
            hist = cv.calcHist([image], [channel], None, [256], [0, 256])
            histograms.append(hist.flatten())  # Flatten the array
    return np.concatenate(histograms)  # Concatenate color histograms



def extract_features(path_directory):
    features = []
    labels = []
    
    # Iterate through the images in the directory
    for filename in os.listdir(path_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv.imread(os.path.join(path_directory, filename))
            
            # Compute the histogram for the image (RGB)
            hist = calculate_histograms(img)
            
            # Add the histogram to the feature list
            features.append(hist)
            
            # The label can be derived from the filename or folder name
            # For example, assume the class is encoded in the filename before the underscore
            class_label = filename.split('_')[0]  # or use another method to obtain the label
            labels.append(class_label)
    
    return np.array(features), np.array(labels)



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


def extract_features_with_autoencoder(autoencoder, data_loader):
    autoencoder.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label_batch in data_loader:
            images = images.view(images.size(0), -1)  # Flatten immagini
            encoded, _ = autoencoder(images)
            features.append(encoded.numpy())
            labels.append(label_batch.numpy())
    return np.vstack(features), np.hstack(labels)


# Sparse Autoencoder
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = torch.relu(self.encoder(x))
        decoded = torch.sigmoid(self.decoder(encoded))
        return encoded, decoded