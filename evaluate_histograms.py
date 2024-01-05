import numpy as np
from detect_skin import detect_skin

def evaluate_histograms(data, skin_hist_rgb, nonskin_hist_rgb, threshold=0.5):
    """
    Evaluates the accuracy of a skin detection model using histograms.

    Args:
        data (numpy.ndarray): The dataset array with shape (N, 4). Each row represents a pixel, and the 
                            columns represent the B, G, R values and the label (1 for skin, 2 for non-skin).
        skin_hist_rgb (np.ndarray): The skin color histogram in RGB format.
        nonskin_hist_rgb (np.ndarray): The non-skin color histogram in RGB format.
        threshold (float, optional): The threshold value for skin detection. Defaults to 0.5.

    Returns:
        tuple: A tuple containing the accuracy, true positives, true negatives, false positives, and false negatives.
    """
    #instantiate variables
    TP = FN = TN = FP = 0
    
    #Fill data into nonskin and skin histograms
    for B, G, R, label in data:
        R, G, B = int(R), int(G), int(B)
        i = R * 32 // 256
        j = G * 32 // 256
        k = B * 32 // 256
        skin_prob = skin_hist_rgb[i, j, k]
        nonskin_prob = nonskin_hist_rgb[i, j, k]
            
        if skin_prob / (skin_prob + nonskin_prob + .000000000000001) > threshold:
            predicted_labels = 1
        else:
            predicted_labels = 2
        
        if (predicted_labels == 1 and label == 1):
            TP +=1
        elif (predicted_labels == 2 and label == 1):
            FN += 1
        elif (predicted_labels == 1 and label == 2):
            FP += 1
        elif (predicted_labels == 2 and label == 2):
            TN += 1
                
    ACC = (TP + TN) / max(TP + TN + FP + FN, 1)

    return ACC, TP, TN, FP, FN
