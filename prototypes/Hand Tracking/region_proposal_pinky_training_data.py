
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

# # coords for training bounding box square
# bounding_box_x = 200
# bounding_box_y = 130

# bounding_box_x_end = 0
# bounding_box_y_end = 0

# bounding_box_number = 0

# for imageNumber in range(29,56):
#     image_filepath = "/Users/mitch/Documents/University/Project/pinky_locator/gesture" + str(imageNumber) + ".jpg"
#     image = cv2.imread(image_filepath)
#     resized = cv2.resize(image, (320,240), interpolation = cv2.INTER_AREA) # resize image to 320 x 240
    
#     fingers_clicked = 0
#     finger_coords = []
#     while(fingers_clicked<5):
#         # end point of bounding box is x + 106 and y + 80
#         bounding_box_x_end = bounding_box_x + 20
#         bounding_box_y_end = bounding_box_y + 20
#         # draw region proposal
#         image = resized.copy()
#         cv2.namedWindow("image")
#         def click_and_save_center_hand(event, x, y, flags, param):
#             global bounding_box_set, fingers_clicked, finger_coords
#             if (event == cv2.EVENT_LBUTTONDOWN):
#                 print(x,y)
#                 finger_coords.append([x,y])
#                 fingers_clicked += 1
#         cv2.setMouseCallback("image", click_and_save_center_hand)
#         cv2.imshow("image", image)
#         color_chosen = False
#         cv2.waitKey(10)  
#     image_filepath = "/Users/mitch/Documents/University/Project/pinky_locator/gesture" + str(imageNumber)
#     np.save(image_filepath, finger_coords)
    

# # Select each region that contains a fingertip by xy coords
for imageNumber in range(1,56):
    block_deltas = []
    coords_filepath = "/Users/mitch/Documents/University/Project/pinky_locator/gesture" + str(imageNumber) + ".npy"
    coords = np.load(coords_filepath)
    if(len(coords)==0):
        print("coords: ", imageNumber, " is empty")
    # coords[0] = x and coords[1] = y
    regions = np.zeros((int(320/5),int(240/5)))
    for r_counter in range(0,320,5):
            for c_counter in range(0,240,5):
                for each_coordinate in coords: # for each region for each pinky coord for each image
                    if(each_coordinate[0] >= r_counter and each_coordinate[0] <= r_counter+20):
                        if(each_coordinate[1] >= c_counter and each_coordinate[1] <= c_counter+20):
                            # coord falls inside region
                            regions[int(r_counter/5)][int(c_counter/5)] = 1
    regions_filepath = "/Users/mitch/Documents/University/Project/pinky_locator/region" + str(imageNumber)
    np.save(regions_filepath, regions)
    fig, ax = plt.subplots(figsize=(15,15))
    sns.heatmap(regions, square=True, ax=ax)
    plt.yticks(rotation=0,fontsize=16)
    plt.xticks(fontsize=12)
    plt.tight_layout()
    plt.show()