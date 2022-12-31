import cv2
import numpy as np
from scipy.signal import convolve2d

# modified edge detection filter
filter = np.array([[-1, -1, -1],
                    [-1, 9, -1],
                    [-1, -1, -1]])

image = cv2.imread("handcoordstrainingdata/gesture0.jpg",0)
output_conv = convolve2d(image, filter, 'valid')
cv2.imshow('Edges',  output_conv.astype(np.float32))
while 1:
    cv2.imshow("hello",image)
    if cv2.waitKey(10) == 27:
        break