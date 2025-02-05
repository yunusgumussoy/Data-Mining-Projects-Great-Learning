# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 16:54:48 2024

@author: Yunus
"""

# OpenCV is a popular library for computer vision tasks.

# pip install opencv-python

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Reading and writing video files
cap = cv.VideoCapture('1.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        cv.imshow('frame', gray)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv.destroyAllWindows()
