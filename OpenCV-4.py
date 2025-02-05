# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:13:09 2024

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

# 1. Image Resizing
resized_img = cv.resize(img, (400, 400), interpolation=cv.INTER_LINEAR)
cv.imshow("Resized Image", resized_img)

# 2. Image Rotation
(h, w) = img.shape[:2]
center = (w // 2, h // 2)
angle = 45
scale = 1.0
rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
rotated_img = cv.warpAffine(img, rotation_matrix, (w, h))
cv.imshow("Rotated Image", rotated_img)

# 3. Affine Transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
affine_matrix = cv.getAffineTransform(pts1, pts2)
affine_transformed_img = cv.warpAffine(img, affine_matrix, (w, h))
cv.imshow("Affine Transformed Image", affine_transformed_img)

# 4. Image Thresholding
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, binary_thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
cv.imshow("Binary Threshold", binary_thresh)

# 5. Contour Detection
contours, _ = cv.findContours(binary_thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contour_img = img.copy()
cv.drawContours(contour_img, contours, -1, (0, 255, 0), 3)
cv.imshow("Contours", contour_img)

# 6. Histogram Calculation and Plotting
colors = ('b', 'g', 'r')
for i, color in enumerate(colors):
    histogram = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histogram, color=color)
    plt.xlim([0, 256])
plt.title('Histogram for color image')
plt.show()

# Wait for a key press indefinitely or for a specified amount of time (0 means indefinitely)
cv.waitKey(0)

# Close all OpenCV windows
cv.destroyAllWindows()
