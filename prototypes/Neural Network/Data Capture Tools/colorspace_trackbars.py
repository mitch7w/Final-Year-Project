import cv2
import numpy as np
import scipy.signal
import sys
import matplotlib.pyplot as plt
import freenect
from PIL import ImageEnhance, Image
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

gestureCounter = 0
def nothing():
    pass
cv2.namedWindow('trackbarWindow')
cv2.createTrackbar('Y1', 'trackbarWindow', 0, 255, nothing)
cv2.createTrackbar('Cb1', 'trackbarWindow', 0, 255, nothing)
cv2.createTrackbar('Cr1', 'trackbarWindow', 0, 255, nothing)
cv2.createTrackbar('Y2', 'trackbarWindow', 0, 255, nothing)
cv2.createTrackbar('Cb2', 'trackbarWindow', 0, 255, nothing)
cv2.createTrackbar('Cr2', 'trackbarWindow', 0, 255, nothing)
Y1 = 0
Cb1 = 0
Cr1 = 0
Y2 =  0
Cb2 = 0
Cr2 = 0
while True:
    # read in from Kinect
    depthInput = freenect.sync_get_depth()[0]
    videoInput = freenect.sync_get_video()[0]
  
    # cv2.imshow('Input Gesture', cv2.flip(image, 1))
    # also show segmentation output
    # increase saturation
    im_pil = Image.fromarray(videoInput)
    converter = ImageEnhance.Color(im_pil)
    img2 = converter.enhance(1.5)
    # For reversing the operation:
    im_np = np.asarray(img2)
    image = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("Enhancer", image)
    # print(image_ycrcb[0][0])
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
    # 44 ,  0 ,  194 ,  130 ,  70 ,  255 HSV ranges
    print(Y1, ", ", Cb1, ", ", Cr1, ", ", Y2, ", ", Cb2, ", ", Cr2)
    # skin1 = (114,74,147)
    # skin2 = (255,190,204)
    skin1 = (Y1, Cb1, Cr1)
    skin2 = (Y2, Cb2, Cr2)
    mask = cv2.inRange(image_ycrcb, skin1, skin2)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('Color mask', mask)
    cv2.imshow('Ycbcr color', image_ycrcb)
    cv2.imshow('Result', result)
    # cv2.imshow('Color Gesture', result)
    # cv2.imwrite("result.png", result)
    greyscale = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(mask, (320,240), interpolation = cv2.INTER_AREA)
    # modified edge detection filter
    filter = np.array([[0, -1, 0],
                      [-1, 4, -1],
                      [0, -1, 0]])
                    # try do edge detection first before color segmenting
    output_conv = scipy.signal.convolve2d(resized, filter, 'valid')
    # cv2.imshow('Segmented Gesture', output_conv.astype(np.uint8))
    # if cv2.waitKey(5) == ord(" "):
    #   image_filepath = "fist_classifier_new_webcam/gesture" + str(gestureCounter) + ".jpg"
    #   cv2.imwrite(image_filepath, image)
    #   gestureCounter +=1
    k = cv2.waitKey(1000) & 0xFF
    if k == 27:
        break
    Y1 = cv2.getTrackbarPos('Y1', 'trackbarWindow')
    Cb1 = cv2.getTrackbarPos('Cb1', 'trackbarWindow')
    Cr1 = cv2.getTrackbarPos('Cr1', 'trackbarWindow')
    Y2 = cv2.getTrackbarPos('Y2', 'trackbarWindow')
    Cb2 = cv2.getTrackbarPos('Cb2', 'trackbarWindow')
    Cr2 = cv2.getTrackbarPos('Cr2', 'trackbarWindow')
  
# cap.release()

# open hand, fist, face, environment

# good skin color range in Ycbcr seems to be 50 ,  89 ,  136 ,  231 ,  147 ,  181