# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:45:51 2024

@author: Yunus
"""

# OpenCV is a popular library for computer vision tasks.

# pip install opencv-python

import cv2 as cv
import numpy as np

path = "C:/Users/yunus/Downloads/1.jpg"

img = cv.imread(path)

# Check if the image was loaded correctly
if img is None:
    print("Error: Could not load image.")
    exit()

# Display the original image
cv.namedWindow("Original Image", cv.WINDOW_AUTOSIZE)
cv.imshow("Original Image", img)

# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Grayscale Image", gray)

# Apply Gaussian blur to the image
blurred = cv.GaussianBlur(gray, (5, 5), 0)
cv.imshow("Blurred Image", blurred)

# Perform Canny edge detection
edges = cv.Canny(blurred, 50, 150)
cv.imshow("Canny Edges", edges)

# Draw a rectangle on the original image
# Parameters: image, top-left corner, bottom-right corner, color, thickness
cv.rectangle(img, (50, 50), (200, 200), (0, 255, 0), 2)

# Draw a circle on the original image
# Parameters: image, center, radius, color, thickness
cv.circle(img, (300, 300), 50, (255, 0, 0), 3)

# Draw a line on the original image
# Parameters: image, start point, end point, color, thickness
cv.line(img, (400, 400), (500, 500), (0, 0, 255), 4)

# Write text on the original image
# Parameters: image, text, bottom-left corner, font, font scale, color, thickness, line type
cv.putText(img, 'OpenCV Demo', (50, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv.LINE_AA)

# Display the image with drawings
cv.imshow("Image with Drawings", img)

# Wait for a key press indefinitely or for a specified amount of time (0 means indefinitely)
cv.waitKey(0)

# Close all OpenCV windows
cv.destroyAllWindows()
