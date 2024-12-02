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
from send2trash import send2trash
import cv2 as cv


def check_dimensions(data):
    expected_dimensions = None

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
                break
        
def check_channels(data):
    expected_channels = None

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
                break
            


def check_max_min_dimensions_and_channels(data):
    # Initialize variables to track min/max values
    min_width = 0
    min_height = 0
    max_width = 0
    max_height = 0
    min_channels = 0
    max_channels = 0

    for idx in range(len(data)):
        image= data[idx][0]
        width, height, num_channels = image.shape[1], image.shape[2], image.shape[0]
                        
        min_width = min(min_width, width)
        max_width = max(max_width, width)
        min_height = min(min_height, height)
        max_height = max(max_height, height)
                        
        # Update min/max for channels
        min_channels = min(min_channels, num_channels)
        max_channels = max(max_channels, num_channels)

    # Output the results
    print(f"Min Width: {min_width}, Max Width: {max_width}")
    print(f"Min Height: {min_height}, Max Height: {max_height}")
    print(f"Min Channels: {min_channels}, Max Channels: {max_channels}")


def count_small_images(data, size_threshold=(128, 128), n_channels = 3):
    # Initialize a dictionary to store the count of small images for each class
    small_images_count = {}
    small_count = 0
    few_channels = 0
    for idx in range(len(data)):
        image, label = data[idx]
        width, height, num_channels = image.shape[1], image.shape[2], image.shape[0]
        if width < size_threshold[0] and height < size_threshold[1]:
            small_count += 1
        if num_channels < n_channels:
            few_channels += 1
            
    # Store the result for this class
    small_images_count[label] = (small_count, few_channels)

    # Output the results for each class
    for class_name, count in small_images_count.items():
        print(f"Class {class_name}: {count[0]} images are smaller than ({size_threshold[0]}x{size_threshold[1]})")
        print(f'Number of images with less than {num_channels} channels in {class_name}: {count[1]}')    


def delete_invalid_images(data, size_threshold=(224, 224), min_channels=3):
    # Iterate in reverse order to avoid index shifting
    for idx in range(len(data) - 1, -1, -1):
        image= data[idx][0]
        width, height, num_channels = image.shape[1], image.shape[2], image.shape[0]
        # Check if the image is smaller than the threshold or has fewer than the required channels
        if (width < size_threshold[0] or height < size_threshold[1]) or num_channels < min_channels:
            # Delete the invalid image
            data.pop(idx)
            print(f"Deleted image (Size: {width}x{height}, Channels: {num_channels})")
        
    print("Invalid images deleted.")




#def resize_and_change_n_channels(folder, size = (224, 224), n_channels = 3):
