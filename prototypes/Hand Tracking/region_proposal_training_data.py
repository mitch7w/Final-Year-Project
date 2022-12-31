
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
from math import sqrt
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

# cap = cv2.VideoCapture(0)

# coords for training bounding box square
bounding_box_x = 0
bounding_box_y = 0

bounding_box_x_end = 0
bounding_box_y_end = 0

bounding_box_number = 0

# coords start 0,0 at top left corner


# label xy training data

for imageNumber in range(558):
    image_filepath = "/Users/mitch/Documents/University/Project/hand_detector_images/gesture" + str(imageNumber) + ".jpg"
    image = cv2.imread(image_filepath)
    resized = cv2.resize(image, (320,240), interpolation = cv2.INTER_AREA) # resize image to 320 x 240
    
    bounding_box_set = False
    while(bounding_box_set == False):
        # end point of bounding box is x + 106 and y + 80
        bounding_box_x_end = bounding_box_x + 106
        bounding_box_y_end = bounding_box_y + 80
        # draw region proposal
        image = resized.copy()
        cv2.namedWindow("image")
        def click_and_save_center_hand(event, x, y, flags, param):
            global bounding_box_set
            if (event == cv2.EVENT_LBUTTONDOWN):
                print(x,y)
                image_filepath = "hand_detector_images/gesture" + str(imageNumber)
                np.save(image_filepath, [x,y])
                bounding_box_set = True
        cv2.setMouseCallback("image", click_and_save_center_hand)
        cv2.imshow("image", cv2.rectangle(image, (bounding_box_x,bounding_box_y), (bounding_box_x_end,bounding_box_y_end), (0,0,255), 2))
        cv2.waitKey(1)

# # center of each blocks
# block_centers = []
# # centre of first block
# x = 53
# y = 40
# for y_count in range(17):
#     for x_count in range(22):
#         block_centers.append((x,y))
#         x += 10
#     y += 10
#     x = 53

print(block_centers)

# # Calculate distances from the centres of each block
# for imageNumber in range(558):
#     block_deltas = []
#     coords_filepath = "/Users/mitch/Documents/University/Project/hand_detector_images/gesture" + str(imageNumber) + ".npy"
#     coords = np.load(coords_filepath)
#     # coords[0] = x and coords[1] = y
#     for block in block_centers:
#         # block[0] = center block x and block[1] = center block y
#         new_delta = sqrt( (block[0]-coords[0])**2 + (block[1]-coords[1])**2 )
#         new_delta_normalized = new_delta / sqrt((240)**2+(320)**2)
#         block_deltas.append(new_delta_normalized)
#     block_delta_filepath = "/Users/mitch/Documents/University/Project/hand_detector_images/block_delta" + str(imageNumber)
#     np.save(block_delta_filepath, block_deltas)

# import numpy as np
# coords_filepath = "/Users/mitch/Documents/University/Project/hand_detector_images/block_delta0.npy"
# delta = np.load(coords_filepath)
# print(delta)
# import numpy as np
# import math
# import matplotlib.pyplot as plt
# import seaborn as sns

# delta = np.array(delta)
# # x_res=x.reshape(math.sqrt(len(x)),math.sqrt(len(x))) #old
# delta_res=delta.reshape((17,22))

# fig, ax = plt.subplots(figsize=(15,15))
# sns.heatmap(delta_res, square=True, ax=ax)
# plt.yticks(rotation=0,fontsize=16)
# plt.xticks(fontsize=12)
# plt.tight_layout()
# plt.show()