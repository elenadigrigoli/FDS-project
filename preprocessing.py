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
import time


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


def count_small_images(data, size_threshold=(128, 128)):
    # Initialize a dictionary to store the count of small images for each class
    small_images_count = {}
    tot_small = 0
    class_map = {k:v for k,v in enumerate(data.classes)}
    for idx in range(len(data)):
        image, label = data[idx]
        width, height = image.shape[1], image.shape[2]

        if label not in small_images_count:
            small_images_count[label] = 0
        
        # Update counts for class
        if width < size_threshold[0] and height < size_threshold[1]:
            small_images_count[label] += 1
            tot_small += 1


    # Output the results for each class
    for label, counts in small_images_count.items():
        class_name = class_map[label]
        print(f"Class {class_name}: {counts} images have at least one dimension smaller than {size_threshold[0]} or {size_threshold[1]}.")
    print(f'Total invalid images: {tot_small}')

def delete_invalid_images(data, size_threshold=(224, 224)):
    
    #valid_data = [image for image in data if image[0].shape[1] >= size_threshold[0] and image[0].shape[2] >= size_threshold[1]]
    
    valid_data = []

    for idx in range(len(data)):
        image = data[idx][0]
        width, height = image.shape[1], image.shape[2]
        # Check if the image is bigger than the threshold
        if width >= size_threshold[0] and height >= size_threshold[1]:
            # Append the valid image
            valid_data.append(data[idx])
        else:
            print(f"Deleted image (Size: {width}x{height}).")
    

        
    print("Invalid images deleted.")
    print(f'{len(data)-len(valid_data)} images removed.')

    return valid_data


#def resize_and_change_n_channels(folder, size = (224, 224), n_channels = 3):
