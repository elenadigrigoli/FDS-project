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



# Function to compute the normalized histogram of an image
def compute_grayscale_histogram(image):
    # Convert the image to grayscale
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Compute the histogram of the grayscale image (pixel intensity values)
    hist = cv.calcHist([image_gray], [0], None, [256], [0, 256])
    
    # Normalize the histogram (if the image is already normalized, this step can be omitted)
    hist = hist / hist.sum()
    
    # Return the histogram as a flattened vector
    return hist.flatten()


# Function to compute the normalized histogram of an image
def compute_histogram(image):
    
    # Compute the histogram of the grayscale image (pixel intensity values)
    hist = cv.calcHist([image], [0], None, [256], [0, 256])
    
    # Normalize the histogram (if the image is already normalized, this step can be omitted)
    hist = hist / hist.sum()
    
    # Return the histogram as a flattened vector
    return hist.flatten()



# Function to extract features from a set of images in a directory
def extract_features(path_directory):
    features = []
    labels = []
    
    # Iterate through the images in the directory
    for filename in os.listdir(path_directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv.imread(os.path.join(path_directory, filename))
            
            # Compute the histogram for the image (RGB)
            hist = compute_histogram(img)
            
            # Add the histogram to the feature list
            features.append(hist)
            
            # The label can be derived from the filename or folder name
            # For example, assume the class is encoded in the filename before the underscore
            class_label = filename.split('_')[0]  # or use another method to obtain the label
            labels.append(class_label)
    
    return np.array(features), np.array(labels)