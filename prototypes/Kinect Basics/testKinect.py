# import freenect
# import numpy as np
# import cv2
# import importlib.util
# spec = importlib.util.spec_from_file_location("frame_convert", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert.py")
# frame_convert = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(frame_convert)
# import matplotlib.pyplot as mp
# import signal
# spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
# frame_convert2 = importlib.util.module_from_spec(spec1)
# spec1.loader.exec_module(frame_convert2)

# def get_depth():
#     # freenect.sync_get_depth() returns (depth, timestamp) or None on error
#         # depth: A numpy array, shape:(480,640) dtype:np.uint16 # rows x cols or height x width
#         # timestamp: int representing the time
#         # uint16 is 0 to 65_535
#     return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0]) # returns An opencv image who's datatype is unspecified


# def get_video():
#     # freenect.sync_get_video returns Returns:
#         # (depth, timestamp) or None on error 
#         # depth: A numpy array, shape:(480, 640, 3) dtype:np.uint8 - but this depth is just RGB values actually # rows x cols or height x width x rgb
#         # timestamp: int representing the time
#     return frame_convert2.video_cv(freenect.sync_get_video()[0]) # An opencv image who's datatype is 1 byte, 3 channel BGR


# while 1:
#     cv2.imshow('Depth', get_depth())
#     cv2.imshow('RGB', get_video())

#!/usr/bin/env python
import freenect
import cv2
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)
spec = importlib.util.spec_from_file_location("frame_convert", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert.py")
frame_convert = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frame_convert)

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')

def get_depth():
    depth_data = freenect.sync_get_depth()[0]
    print("depth_data: ", depth_data)
    return frame_convert2.pretty_depth_cv(depth_data)

def get_video():
    video_data = freenect.sync_get_video()[0]
    return frame_convert2.video_cv(video_data)

while 1:
    cv2.imshow('Depth', get_depth())
    cv2.imshow('Video', get_video())
    if cv2.waitKey(10) == 27:
        break