import os
import numpy as np
import cv2 as cv

# Function to load images from a directory
def load_images(root_dir, image_size=(256, 256)):
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
                img = cv.imread(image_path)
                if img is not None:  # Ensure image is valid
                    img = cv.resize(img, image_size)
                    images.append(img)
                    labels.append(class_index)  # Label as the folder index
    return np.array(images), np.array(labels), class_names