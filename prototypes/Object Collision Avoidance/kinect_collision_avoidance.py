import cv2
import numpy as np
import freenect
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)

def check_space_occupied(depth_data, proposed_x, proposed_y, proposed_z):
    # check depthInput at pixels to see if object there
    depth_x = int((proposed_x/8)*640)
    depth_y = int((proposed_y/6)*480)
    depth_z = int(650 -10*(proposed_z-3))
    for x in range(depth_x-20, depth_x+20):
        for y in range(depth_y-20, depth_y+20):
            if(depth_x >=0 and depth_x < 640 and depth_y >= 0 and depth_y < 480):
                if( depth_data[depth_x][depth_y] >= depth_z-50 and depth_data[depth_x][depth_y] <= depth_z+50):
                    return True
    return False

while True:
    # get Kinect data
    cv2.namedWindow("Kinect Input")
    depthInput = freenect.sync_get_depth()[0]
    videoInput = freenect.sync_get_video()[0]
    videoFeed = frame_convert2.video_cv(videoInput) # videoFeed is 480 x 640 x 3
    # depthInput.shape = (480,640)
    print(depthInput[0][0])
    cv2.imshow("Kinect Input", videoFeed)
    cv2.waitKey(1)

# world coordinate system
# have cube at specific xy coord and z value which corresponds to certain depth value - do manually by moving hand back where cube would be and checking z value at that point

