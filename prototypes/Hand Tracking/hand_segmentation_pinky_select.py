##########################################
# YCBCR COLOR SPACE
##########################################

import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize)

while True:
    # for imageNumber in range(1,56):
    for imageNumber in range(666,667):
        image_pathname = "/Users/mitch/Documents/University/Project/pinky_locator/gesture" + str(imageNumber) + ".jpg"
        image = cv2.imread(image_pathname)
        image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
        # print(image_ycrcb[0][0])
        skin1 = (50, 89, 136)
        skin2 = (231, 147, 181)
        mask = cv2.inRange(image_ycrcb, skin1, skin2)
        result = cv2.bitwise_and(image, image, mask=mask)
        cv2.imshow("Masked Image", result)

        # try countouring
        # get threshold image
        ret, thresh = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)
        cv2.imshow("Mask", mask)
        # draw the contours on the empty image
        contours = []
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if(len(contours) != 0):
            contours = max(contours, key=lambda x: cv2.contourArea(x))
            empty_image = np.zeros((960, 1280, 3))    
            cv2.drawContours(empty_image, [contours], -1, (255, 255, 0), 2)
            ############# Get edge hand coords ###################
            # hand_points_selected = 0
            # xy_coords = []
            # while(hand_points_selected <2):
            #     def click_and_save_center_hand(event, x, y, flags, param):
            #         global hand_points_selected
            #         if (event == cv2.EVENT_LBUTTONDOWN):
            #             hand_points_selected +=1
            #             print(x,y)
            #             xy_coords.append([x,y])
            #     cv2.setMouseCallback("contours", click_and_save_center_hand)
            #     cv2.imshow("contours", empty_image)
            #     if cv2.waitKey(5) == ord(" "):
            #         break    
            # xy_filepath = "/Users/mitch/Documents/University/Project/pinky_locator/gesture" + str(imageNumber)
            # np.save(xy_filepath, xy_coords)
            ############################################################
            # print(contours)
            # coords_filepath = "/Users/mitch/Documents/University/Project/pinky_locator/gesture" + str(imageNumber) + ".npy"
            # xy_coords = np.load(coords_filepath)
            x = contours[:,:,0]
            y = contours[:,:,1]
            fig, axs = plt.subplots(3)
            axs[0].plot(x)
            axs[0].set_ylabel("X"),
            axs[1].plot(y)
            axs[1].set_ylabel("Y")
            axs[2].plot(x,y)
            # axs[2].plot(xy_coords[0][0],xy_coords[0][1], "r+")
            # axs[2].plot(xy_coords[1][0],xy_coords[1][1], "r+")
            axs[2].set_xlabel("X")
            axs[2].set_ylabel("Y")
            plt.show()
            
        
cap.release()