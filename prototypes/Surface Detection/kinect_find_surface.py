import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import freenect
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)

while True:
    # read in from Kinect
    videoInput = freenect.sync_get_video()[0]
    depthInput = freenect.sync_get_depth()[0]
    # depthInput[0-480][0-640] # top to bottom, left to right
    # (480, 640)

    # # Perpendicular surface detection with sliding windows
    # # surfaces is array whether or not a surface is present there
    window_averages = []
    distance_from_average = np.zeros(480*640).reshape(depthInput.shape)
    for row in range(0,460,40): # window is 20 high
        for col in range(0,620,40): # window is 20 wide
            # get average depth value of window
            window_average = np.mean(depthInput[row:row+5,col:col+5])
            window_averages.append(window_average)
            # find deviation from average for each pixel in window
            for r in range(40):
                for c in range(40):
                    distance_from_average[row+r][col+c] = abs(depthInput[row+r][col+c] - window_average)
    # plot heatmap for deviations
    print("Plot")   
    print(window_averages) 
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(distance_from_average, square=True, ax=ax)
    plt.tight_layout()
    plt.show()

    # # Row surface detection
    # # surfaces is array whether or not a surface is present there
    # window_averages = []
    # distance_from_average = np.zeros(480*640).reshape(depthInput.shape)
    # for row in range(0,460): # for each row
    #     row_average = np.mean(depthInput[row])
    #     for col in range(0,620): # for each col
    #         # find deviation from average for each pixel in window
    #         # distance_from_average[row][col] = abs(depthInput[row][col] - row_average)
    #         print("abs(depthInput[row][col] - row_average): ", abs(depthInput[row][col] - row_average))
    #         if(abs(depthInput[row][col] - row_average) <= 5): # if close to average of row
    #             distance_from_average[row][col] = 1
    #         else:
    #             distance_from_average[row][col] = 0
    # # plot heatmap for deviations
    # print("Plot")   
    # print(window_averages) 
    # fig, ax = plt.subplots(figsize=(15,15))
    # sns.heatmap(distance_from_average, square=True, ax=ax)
    # plt.tight_layout()
    # plt.show()