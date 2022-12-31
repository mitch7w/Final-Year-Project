import cv2
import numpy as np
import freenect
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)

# Webcam

# gestureCounter = 141
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#   success, image = cap.read()
#   if not success:
#     print("Ignoring empty camera frame.")
#     # If loading a video, use 'break' instead of 'continue'.
#     continue
#   cv2.imshow("Image", image)
#   if cv2.waitKey(5) == ord(" "):
#     image_filepath = "/Users/mitch/Documents/University/Project/topdown_directions/gesture" + str(gestureCounter) + ".jpg"
    
#     cv2.imwrite(image_filepath, image)
#     gestureCounter +=1
# cap.release()

################################################################################################
################################################################################################

# # Kinect press space to get images

# gesture_counter = 779
# while True:
  
#   # get Kinect data
#   cv2.namedWindow("Kinect Input")
#   depthInput = freenect.sync_get_depth()[0]
#   videoInput = freenect.sync_get_video()[0]
#   videoFeed = frame_convert2.video_cv(videoInput) # videoFeed is 480 x 640 x 3
#   # videoFeed[0:100,500:600] access part of videoFeed like such
#   cv2.imshow("Kinect Input", videoFeed)
#   if cv2.waitKey(5) == ord(" "):
#     image_filepath = "/Users/mitch/Documents/University/Project/kinect_side_open_fist_massive/gesture" + str(gesture_counter) + ".jpg"
#     cv2.imwrite(image_filepath, videoFeed)
#     gesture_counter +=1

# Kinect 5 pics a second

gesture_counter = 701

while True:
  
  # get Kinect data
  cv2.namedWindow("Kinect Input")
  depthInput = freenect.sync_get_depth()[0]
  videoInput = freenect.sync_get_video()[0]
  videoFeed = frame_convert2.video_cv(videoInput) # videoFeed is 480 x 640 x 3
  # videoFeed[0:100,500:600] access part of videoFeed like such
  cv2.imshow("Kinect Input", videoFeed)
  image_filepath = "/Users/mitch/Documents/University/Project/open_fist_far/gesture" + str(gesture_counter) + ".jpg"
  cv2.imwrite(image_filepath, videoFeed)
  print("Saved image # ", gesture_counter, end="\r")
  gesture_counter +=1
  cv2.waitKey(200) 
