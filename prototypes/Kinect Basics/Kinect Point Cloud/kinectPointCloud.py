from locale import currency
import freenect
import cv2
import frame_convert2
import numpy as np
import matplotlib.pyplot as plt

cv2.namedWindow('Depth')
cv2.namedWindow('Video')
print('Press ESC in window to stop')

def drawCloud(pCloud):
    for row in range(len(pCloud)):
        for col in range(len(pCloud[row])):
            pCloud[row][col] = pCloud[row][col] / 2047
    pCloud = np.array(pCloud)
    plt.imshow(pCloud, cmap='tab20c', interpolation='none')
    plt.show()

def get_depth():
    currentDepthArray = freenect.sync_get_depth()[0]
    # drawCloud(currentDepthArray)
    # save array into textfile
    # with open("pointCloud.txt", "w") as txt_file:
    #     for row in currentDepthArray:
    #         for col in row:
    #             txt_file.write(str(col) + ",")
    #         txt_file.write("\n")
    # exit()
    return frame_convert2.pretty_depth_cv(currentDepthArray)


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])


while 1:
    cv2.imshow('Depth', get_depth())
    cv2.imshow('Video', get_video())
    if cv2.waitKey(10) == 27:
        break
