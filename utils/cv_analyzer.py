import cv2
import numpy as np

def calculate_decay_index(img_pil):
    # PIL image ko OpenCV RGB format me convert karna
    img_np = np.array(img_pil)
    
    # RGB se HSV color space me shift karna
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Rot/decay (bhure aur kaale dhabbe) ke liye color thresholding
    lower_rot = np.array([10, 40, 20])
    upper_rot = np.array([30, 255, 200])
    
    # Masking surface decay percentage calculation
    mask = cv2.inRange(hsv, lower_rot, upper_rot)
    rot_pixels = np.count_nonzero(mask)
    total_pixels = mask.shape[0] * mask.shape[1]
    
    decay_percentage = (rot_pixels / total_pixels) * 100
    return round(decay_percentage, 2)