# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 15:25:28 2024

@author: Yunus
"""
# OpenCV is a popular library for computer vision tasks.

# pip install opencv-python


import cv2 as cv

path = "C:/Users/yunus/Downloads/1.jpg"

img = cv.imread(path)

print(type(img))

# cv.namedWindow() creates a window with a specified name ("opencv_test" in this case) where the image will be displayed.
# namedWindow
cv.namedWindow("opencv_test", cv.WINDOW_AUTOSIZE)

# cv.WINDOW_AUTOSIZE means the window size will automatically adjust to fit the image size.
# imshow
cv.imshow("opencv_test", img)

# cv.waitKey() waits for a key event for a specified amount of time (in milliseconds).
cv.waitKey(1)

#cvtColor
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray", gray)
cv.waitKey(1)

#imwrite
cv.imwrite("gray_1.jpeg", gray)

# to open the original image as grey
img = cv.imread(path, cv.IMREAD_GRAYSCALE)
cv.namedWindow("gray", cv.WINDOW_AUTOSIZE)
cv.imshow("gray", gray)
cv.waitKey(1)

# all the OpenCV windows are closed
# cv.destroyAllWindows()

