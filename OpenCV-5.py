# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:18:32 2024

@author: Yunus
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load an image from file
path = "C:/Users/yunus/Downloads/3.jpg"
img = cv.imread(path)

# Check if the image was loaded correctly
if img is None:
    print("Error: Could not load image.")
    exit()

# Display the original image
cv.imshow("Original Image", img)

# 1. Image Segmentation using Watershed Algorithm
# Convert to grayscale and apply threshold
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
_, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

# Noise removal using morphological operations
kernel = np.ones((3, 3), np.uint8)
opening = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=2)

# Sure background area
sure_bg = cv.dilate(opening, kernel, iterations=3)

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
_, sure_fg = cv.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg, sure_fg)

# Marker labeling
_, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers + 1

# Mark the unknown region with zero
markers[unknown == 0] = 0

# Apply the Watershed algorithm
markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]

cv.imshow("Segmented Image", img)

# 2. Object Detection using Haar Cascades
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw rectangles around detected faces
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv.imshow("Detected Faces", img)

# 3. Feature Detection using ORB (Oriented FAST and Rotated BRIEF)
# Create ORB detector
orb = cv.ORB_create()

# Detect keypoints and compute descriptors
keypoints, descriptors = orb.detectAndCompute(img, None)

# Draw keypoints
img_with_keypoints = cv.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)
cv.imshow("ORB Keypoints", img_with_keypoints)

# 4. Perspective Transformation
# Define points for perspective transformation
pts1 = np.float32([[50, 50], [200, 50], [50, 200], [200, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250], [200, 200]])

# Compute the perspective transformation matrix
perspective_matrix = cv.getPerspectiveTransform(pts1, pts2)

# Apply perspective transformation
perspective_transformed_img = cv.warpPerspective(img, perspective_matrix, (img.shape[1], img.shape[0]))
cv.imshow("Perspective Transformation", perspective_transformed_img)

# Wait for a key press indefinitely or for a specified amount of time (0 means indefinitely)
cv.waitKey(0)

# Close all OpenCV windows
cv.destroyAllWindows()
