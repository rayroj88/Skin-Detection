import numpy as np
import cv2
from detect_skin import detect_skin

def build_histograms(data):
    """
    Builds skin and non-skin color histograms from a given dataset file.

    Args:
    - data (numpy.ndarray): The dataset array with shape (N, 4). Each row represents a pixel, and the 
                            columns represent the B, G, R values and the label (1 for skin, 2 for non-skin).

    Returns:
    - skin_histogram (numpy.ndarray): A 3D numpy array representing the skin color histogram.
    - nonskin_histogram (numpy.ndarray): A 3D numpy array representing the non-skin color histogram.
    """
    
    histogram_bins = data.shape[0]

    
    skin_histogram = np.zeros((32, 32, 32))
    nonskin_histogram = np.zeros((32, 32, 32))
    
    for B, G, R, label in data:
        R, G, B = int(R), int(G), int(B)
        i = R * 32 // 256
        j = G * 32 // 256
        k = B * 32 // 256
        if (label == 1):
            skin_histogram[i, j, k] +=1
        else:
            nonskin_histogram[i, j, k] += 1  
    total_skin_pixels = np.sum(skin_histogram)
    total_nonskin_pixels = np.sum(nonskin_histogram)
    
    skin_histogram /= total_skin_pixels
    nonskin_histogram /= total_nonskin_pixels
         
    return skin_histogram, nonskin_histogram