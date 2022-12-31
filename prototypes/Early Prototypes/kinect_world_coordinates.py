# Find edges of objects in kinect camera by utilizing image conv. filters

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.signal import convolve2d
import freenect
import cv2
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize) # makes arrays print out in full


cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')
while 1:
    depthInput = freenect.sync_get_depth()[0]
    print(depthInput[0][0])
    videoInput = freenect.sync_get_video()[0]
    # segment based on closest value
    # depthInput = depthInput[ (depthInput >= np.amin(depthInput)) ]
    # depthInput = np.clip(depthInput, np.amin(depthInput),np.amin(depthInput)+200) # clip depth Input based on min and mix values here
    # depthInput = np.minimum(depthInput, np.amin(depthInput))
    
    # result = np.where(depthInput == np.amin(depthInput)) # return index of min element
    # print('Minimum element from depthInput : ', np.amin(depthInput))
    videoFeed = frame_convert2.video_cv(videoInput)
    cv2.imshow('videoFeed', videoFeed)
    depthFeed = frame_convert2.pretty_depth_cv(depthInput)
    depthFeedColoured = cv2.applyColorMap(depthFeed,cv2.COLORMAP_JET)
    cv2.imshow('depthFeed', depthFeedColoured)
    if cv2.waitKey(10) == 27:
        break


# create world coordinates
# fill world coordinates with kinect data where objects are (either 0 or 1 if depth data present at this point)
# try move cube through world
