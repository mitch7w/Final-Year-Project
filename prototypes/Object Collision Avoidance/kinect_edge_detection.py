# Find edges of objects in kinect camera by utilizing image conv. filters

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

person = imread('Neural network/person.PNG')

# coordinates for current pixel depth
current_pixel_x = 0
current_pixel_y = 0

# top left coordinates of virtual cube 10px by 10px
cubeX = 0
cubeY = 0
#cubeZ later

# check if current pixel occupied
def pixelOccupied(currentDepthFeed, x,y,z):
    if currentDepthFeed[x][y] > z : # camera current reports that depth pixel as being unoccupied
        return False
    else: 
        return True

def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:,:,0], kernel, 'valid')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    blue = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)

# modified edge detection filter
filter = np.array([[-1, -1, -1],
                    [-1, 10, -1],
                    [-1, -1, -1]])


cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')

# conv_im1 = rgb_convolve2d(person, identity)
# fig, ax = plt.subplots(1,2, figsize=(12,5))
# ax[0].imshow(identity, cmap='gray')
# ax[1].imshow(abs(conv_im1), cmap='gray')
# plt.show()

while 1:
    depthInput = freenect.sync_get_depth()[0]
    videoInput = freenect.sync_get_video()[0]
    # depthText = "depthInput: " + str(depthInput)
    # kinect_output_file = open("kinect_output_file.txt", "w+")
    # kinect_output_file.write(str(depthText)+"\n\n")
    # kinect_output_file.close()
    # exit()
    
    
    depthFeed = frame_convert2.pretty_depth_cv(depthInput)
    depthFeedColoured = cv2.applyColorMap(depthFeed,cv2.COLORMAP_JET)
    # depth data goes from 0 - 2047
    # segment background
    
    for x in range(len(depthInput)):
        for y in range(len(depthInput[x])):
            if(depthInput[x][y] > 150):
                videoInput[x][y] = [0,0,0]
    videoFeed = frame_convert2.video_cv(videoInput)
    cv2.imshow('videoFeed', videoFeed)                
    cv2.imshow('depthFeed', depthFeedColoured)
    # print(depthFeed)
    # print("Pixel 0 occupied: ", pixelOccupied(depthInput,0,0,0), depthInput[0][0])
    

    # Edge detection
    # conv_im1 = rgb_convolve2d(videoInput, filter)
    # # cv2.imshow('Edges',  np.array(newIm))
    # conv_im1[current_pixel_x][current_pixel_y] = 0
    # cv2.imshow('Edges',  conv_im1.astype(np.float32))

    # print("Current pixel ", current_pixel_x, " ", current_pixel_y, ": ", depthInput[current_pixel_x][current_pixel_y])
    if cv2.waitKey(5) == ord('i'):
        current_pixel_x +=1
    if cv2.waitKey(5) == ord('o'):
        current_pixel_y +=1


# not 255 = in frame