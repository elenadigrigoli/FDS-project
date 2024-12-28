import os
import math
import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import random_split, DataLoader
import tensorflow as tf
from skimage import io, color, transform
import sklearn
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2 as cv
import shutil
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




def count_images_per_class(data):
    labels = [label for _, label in data]
    
    class_counts = Counter(labels)
    
    for class_idx, count in class_counts.items():
        class_name = data.classes[class_idx]
        print(f"Class {class_name}: {count} images")



def augment_images(class_name, input_class_dir, current_count, target_count, augmentations):
    # Generate images until target size
    augmentation_needed = target_count - current_count
    print(f"Augmenting {augmentation_needed} images for class: {class_name}")

    images = [os.path.join(input_class_dir, img) for img in os.listdir(input_class_dir) if img.endswith(('.jpg', '.jpeg', '.png'))]

    for i in range(augmentation_needed):
        # Choose random image for class
        img_path = images[i % len(images)]
        img = Image.open(img_path).convert("RGB")
        transformed_image = augmentations(img)

        # Save augmented image in class folder
        augmented_img_name = f"aug_{i}.jpg"
        augmented_img_path = os.path.join(input_class_dir, augmented_img_name)
        transformed_image.save(augmented_img_path)



def split_dataset(original_dir, train_dir, test_dir, test_size=0.2, random_state=42):
    for class_name in os.listdir(original_dir):
        class_path = os.path.join(original_dir, class_name)
        
        # Ensure it is a directory
        if not os.path.isdir(class_path):
            continue
        
        # Get all file paths for the current class
        files = [os.path.join(class_path, f) for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        
        # Split into train and test
        train_files, test_files = train_test_split(files, test_size=test_size, random_state=random_state)
        
        # Create subdirectories for each class in train/test
        train_class_dir = os.path.join(train_dir, class_name)
        test_class_dir = os.path.join(test_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir, exist_ok=True)
        
        # Copy files to the respective directories
        for file in train_files:
            shutil.copy(file, train_class_dir)
        for file in test_files:
            shutil.copy(file, test_class_dir)
        print(f"Class {class_name} -> Train: {len(train_files)} | Test: {len(test_files)}")