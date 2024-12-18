import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
import tensorflow as tf
from skimage import io, color, transform
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, top_k_accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from PIL import Image
import cv2 as cv
from collections import Counter




def check_dimensions(data):
    expected_dimensions = None
    all_same = True

    for idx in range(len(data)):
        image= data[idx][0]  # Get the image
        
        # Check the shape of the current image
        image_shape = image.shape[1:3]
        
        # If reference_shape is None, set it to the shape of the first image
        if expected_dimensions is None:
            expected_dimensions = image_shape
        
        else:
            # Check if the current image's shape matches the reference
            if image_shape != expected_dimensions:
                print(f"Not all images have the same shape.")
                all_same = False
                break

    if all_same:
        print(f'All images have the same shape: {expected_dimensions}.')




def check_channels(data):
    expected_channels = None
    all_same = True

    for idx in range(len(data)):
        image = data[idx][0]  # Get the image
        # Check the shape of the current image
        n_channels = image.shape[0]
        
        # If reference_shape is None, set it to the shape of the first image
        if expected_channels is None:
            expected_channels = n_channels
        
        else:
            # Check if the current image's shape matches the reference
            if n_channels != expected_channels:
                print(f"Not all images have the same number of channels.")
                all_same = False
                break
    if all_same:
        print(f'All images have {expected_channels} channels.')




def check_max_min_dimensions(data):
    # Initialize variables to track min/max values
    min_width = float('inf')
    min_height = float('inf')
    max_width = float('-inf')
    max_height = float('-inf')

    for idx in range(len(data)):
        image= data[idx][0]
        width, height = image.shape[1], image.shape[2]
                        
        min_width = min(min_width, width)
        max_width = max(max_width, width)
        min_height = min(min_height, height)
        max_height = max(max_height, height)

    # Output the results
    print(f"Min Width: {min_width}, Max Width: {max_width}")
    print(f"Min Height: {min_height}, Max Height: {max_height}")




def plot_image_size_histograms(data, bins=100):
    heights = []
    widths = []

    for idx in range(len(data)):
        image, _ = data[idx]
        _, height, width = image.shape
        heights.append(height)
        widths.append(width)
    
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(heights, bins=bins, color='blue', alpha=0.7)
    plt.title('Histogram of Image Heights')
    plt.xlabel('Height (pixels)')
    plt.ylabel('Frequency')
    plt.xlim(0, 2000)

    plt.subplot(1, 2, 2)
    plt.hist(widths, bins=bins, color='green', alpha=0.7)
    plt.title('Histogram of Image Widths')
    plt.xlabel('Width (pixels)')
    plt.ylabel('Frequency')
    plt.xlim(0, 2000)


    plt.tight_layout()
    plt.show()



def count_images_per_class(data):
    labels = [label for _, label in data]
    
    class_counts = Counter(labels)
    
    for class_idx, count in class_counts.items():
        class_name = data.classes[class_idx]
        print(f"Class {class_name}: {count} images")



def augment_images(class_name, input_class_dir, current_count, target_count, augmentations):
    # Generare immagini augmentate fino a raggiungere il target
    augmentation_needed = target_count - current_count
    print(f"Augmenting {augmentation_needed} images for class: {class_name}")

    images = [os.path.join(input_class_dir, img) for img in os.listdir(input_class_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

    for i in range(augmentation_needed):
        # Scegliere un'immagine casuale dalla classe
        img_path = images[i % len(images)]
        img = Image.open(img_path).convert("RGB")
        transformed_image = augmentations(img)

        # Salvare l'immagine augmentata nella stessa cartella della classe
        augmented_img_name = f"aug_{i}.jpg"
        augmented_img_path = os.path.join(input_class_dir, augmented_img_name)
        transformed_image.save(augmented_img_path)


def compute_histogram(image):
    """
    Compute the color histogram of an image (3 channels: R, G, B).
    Args:
        image (Tensor): A tensor of shape (C, H, W) where C = 3 (RGB channels).
    Returns:
        histograms (list): List of 3 histograms for each channel.
    """
    channels = image.view(3, -1)
    histograms = []
    
    for channel in channels:
        hist = torch.histc(channel, bins=256, min=0, max=1)
        histograms.append(hist)
    
    return histograms

def split_set(dataset):
    x_train = []
    y_train = []

    for data_point in dataset:
        features, labels = data_point  # unpack the tuple
        x_train.append(features)
        y_train.append(torch.tensor(labels))

    # Optionally, convert the lists to tensors if needed
    x_train = torch.stack(x_train)  # Convert to tensor
    y_train = torch.stack(y_train)  # Convert to tensor

    return x_train, y_train


def extract_data_from_set(dataset):
    imgs = []
    labels = []
    for data in dataset:
        inputs, targets = data
        imgs.append(inputs.numpy())  # Convert to numpy array
        labels.append(targets)
    return np.concatenate(imgs), np.concatenate(labels).astype(int)