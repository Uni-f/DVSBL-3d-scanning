import numpy as np

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt(((point1[0] - point2[0]) ** 2) + ((point1[1] - point2[1]) ** 2))

def pixels_to_cm(pixels, pixels_per_cm):
    """Convert pixels to centimeters"""
    return pixels / pixels_per_cm
