# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:59:56 2024

@author: Yunus
"""

import cv2 as cv
import numpy as np

# Load the pre-trained Haar Cascade for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Capture video from webcam
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Edge detection
    edges = cv.Canny(gray, 100, 200)

    # ORB feature detection
    orb = cv.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    frame_with_keypoints = cv.drawKeypoints(frame, keypoints, None, color=(0, 255, 0), flags=0)

    # Face detection
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv.rectangle(frame_with_keypoints, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the results
    cv.imshow('Original', frame)
    cv.imshow('Edges', edges)
    cv.imshow('ORB Keypoints and Faces', frame_with_keypoints)

    # Press 'q' to exit
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
