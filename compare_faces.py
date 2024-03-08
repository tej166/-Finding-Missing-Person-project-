import numpy as np
import cv2
from sklearn.metrics.pairwise import euclidean_distances

def extract_features(image_path, size=(96, 96)):
    """Extracts HOG features from an image.

    Args:
        image_path (str): Path to the image file.
        size (tuple): The size to which the image should be resized before feature extraction.

    Returns:
        np.ndarray: The extracted HOG features.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, size)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = cv2.HOGDescriptor((8, 8), (4, 4), (4, 4), (4, 4), 9).compute(gray_image)
    features = features / np.prod(gray_image.shape[:2])
    return features

def compare_faces(features1, features2, threshold=0.5):
    """Compares two sets of HOG features and returns True if they are similar.

    Args:
        features1 (np.ndarray): The first set of HOG features.
        features2 (np.ndarray): The second set of HOG features.
        threshold (float): The similarity threshold.

    Returns:
        bool: True if the features are similar, False otherwise.
    """
    distance = euclidean_distances(features1.reshape(1, -1), features2.reshape(1, -1))
    return distance < threshold