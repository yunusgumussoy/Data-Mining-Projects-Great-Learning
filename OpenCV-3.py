# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:29:08 2024

@author: Yunus
"""

# OpenCV is a popular library for computer vision tasks.

# pip install opencv-python

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load an image from file
path = "C:/Users/yunus/Downloads/1.jpg"
img = cv.imread(path)

# Check if the image was loaded correctly
if img is None:
    print("Error: Could not load image.")
    exit()

# Display the original image
cv.imshow("Original Image", img)

# Normalize the image using Min-Max normalization
# Convert the image to float32 for more precision
img_float = np.float32(img)

# Calculate the min and max pixel values
min_val = np.min(img_float)
max_val = np.max(img_float)

# Normalize the image to the range [0, 1]
normalized_img = (img_float - min_val) / (max_val - min_val)

# Normalize the image to the range [0, 255] for display
normalized_img = np.uint8(normalized_img * 255)

# Display the normalized image
cv.imshow("Normalized Image", normalized_img)

# Wait for a key press indefinitely or for a specified amount of time (0 means indefinitely)
cv.waitKey(0)

# Close all OpenCV windows
cv.destroyAllWindows()
