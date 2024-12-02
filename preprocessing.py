import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
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


def load_images_from_folder(folder):
    images = []
    labels = []
    data = {}
    # Scorri tutte le sottocartelle (classi)
    for class_name in os.listdir(folder):
        class_folder = os.path.join(folder, class_name)
        if os.path.isdir(class_folder):
            # Scorri tutti i file nella sottocartella
            for filename in os.listdir(class_folder):
                img_path = os.path.join(class_folder, filename)
                try:
                    # Apre l'immagine
                    img = Image.open(img_path)
                    images.append(img)
                    labels.append(class_name)
                except Exception as e:
                    print(f"Can't upload {filename}: {e}")
    for image, label in images, labels:
        data[image] = label
    return data



def check_dimensions_and_channels(folder):
    expected_dimensions = None
    expected_mode = None
    all_same_dim = True
    all_same_mode = True
    # Loop through all the folders and images in the folder
    for class_folder in os.listdir(folder):
        class_folder_path = os.path.join(folder, class_folder)
            
        if os.path.isdir(class_folder_path):
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                    
                try:
                    # Open the image
                    with Image.open(image_path) as img:
                        # Get image dimensions (width, height)
                        img_dimensions = img.size
                        img_channels = img.mode
                            
                        # Check if this is the first image or if dimensions match
                        if expected_dimensions is None:
                            expected_dimensions = img_dimensions
                        elif img_dimensions != expected_dimensions:
                            all_same_dim = False
                        # Check if this is the first image or if mode match
                        if expected_mode is None:
                            expected_mode = img_channels
                        elif img_channels != expected_mode:
                            all_same_mode = False
                            
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
        
    # Final output
    if all_same_dim:
        print("All images have the same dimensions.")
    else:
        print("Not all images have the same dimensions.")

    if all_same_mode:
        print(f'All images have the same number of channels: {expected_mode}.')
    else:
        print('Not all images have the same number of channels.')




def check_max_min_dimensions_and_channels(folder):
    # Initialize variables to track min/max values
    min_width = min_height = float('inf')
    max_width = max_height = 0
    min_channels = float('inf')
    max_channels = 0

    # Traverse through all the folders and files in the base folder
    for class_folder in os.listdir(folder):
        class_folder_path = os.path.join(folder, class_folder)
        
        if os.path.isdir(class_folder_path):
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                
                try:
                    # Open the image
                    with Image.open(image_path) as img:
                        # Get image dimensions (width, height)
                        width, height = img.size
                        
                        # Get the number of channels (1 for grayscale, 3 for RGB, 4 for RGBA)
                        num_channels = len(img.getbands())
                        
                        # Update min/max for dimensions
                        min_width = min(min_width, width)
                        max_width = max(max_width, width)
                        min_height = min(min_height, height)
                        max_height = max(max_height, height)
                        
                        # Update min/max for channels
                        min_channels = min(min_channels, num_channels)
                        max_channels = max(max_channels, num_channels)

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
    
    # Output the results
    print(f"Min Width: {min_width}, Max Width: {max_width}")
    print(f"Min Height: {min_height}, Max Height: {max_height}")
    print(f"Min Channels: {min_channels}, Max Channels: {max_channels}")


def count_small_images(folder, size_threshold=(128, 128), n_channels = 3):
    # Initialize a dictionary to store the count of small images for each class
    small_images_count = {}

    # Traverse through all the folders and files in the base folder
    for class_folder in os.listdir(folder):
        class_folder_path = os.path.join(folder, class_folder)
        
        if os.path.isdir(class_folder_path):
            small_count = 0  # Counter for images smaller than the threshold in this class
            few_channels = 0
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                
                try:
                    # Open the image
                    with Image.open(image_path) as img:
                        # Get image dimensions (width, height)
                        width, height = img.size
                        num_channels = len(img.getbands())
                        # Check if the image is smaller than the threshold
                        if width < size_threshold[0] and height < size_threshold[1]:
                            small_count += 1
                        if num_channels < n_channels:
                            few_channels += 1

                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
            
            # Store the result for this class
            small_images_count[class_folder] = (small_count, few_channels)

    # Output the results for each class
    for class_name, count in small_images_count.items():
        print(f"Class {class_name}: {count[0]} images are smaller than ({size_threshold[0]}x{size_threshold[1]})")
        print(f'Number of images with less than 3 channels in {class_name}: {count[1]}')    


def delete_invalid_images(folder, size_threshold=(224, 224), min_channels=3):
    # Loop through all the folders and files in the base folder
    for class_folder in os.listdir(folder):
        class_folder_path = os.path.join(folder, class_folder)
        
        if os.path.isdir(class_folder_path):
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                
                try:
                    # Open the image
                    with cv.imread(image_path) as img:
                        # Get image dimensions (width, height)
                        width, height = img.size
                        
                        # Get the number of channels (1 for grayscale, 3 for RGB, 4 for RGBA)
                        num_channels = len(img.getbands())
                        
                        # Check if the image is smaller than the threshold or has fewer than 3 channels
                        if (width < size_threshold[0] or height < size_threshold[1]) or num_channels < min_channels:
                            # If the image is invalid, delete it
                            try:
                                os.remove(image_path)
                                print(f"Deleted {image_file} from class {class_folder} (Size: {width}x{height}, Channels: {num_channels})")
                            except FileNotFoundError: 
                                print(f"File '{image_file}' not found.")
                
                except Exception as e:
                    print(f"Error processing {image_file}: {e}")
                    
    print("Invalid images deleted.")


#def resize_and_change_n_channels(folder, size = (224, 224), n_channels = 3):
