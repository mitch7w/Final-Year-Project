import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    image = cv2.resize(image, (320,240), interpolation = cv2.INTER_AREA)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
    # print(image_ycrcb[0][0])
    skin1 = (50, 89, 136)
    skin2 = (231, 147, 181)
    mask = cv2.inRange(image_ycrcb, skin1, skin2)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Color", result)
    cv2.imshow("Mask", mask) # 240 x 320 white or black image

    # find highest concentration of values in mask matrix closest to bottom of image
    r_region_counters = []
    c_region_counters = []
    region_sums = []
    south_deltas = []
    # find all regions with pixel count larger than 270 000
    for r_counter in range(0,320,5):
        for c_counter in range(0,240,5):
            region_sum = np.sum(mask[r_counter:r_counter+60,c_counter:c_counter+80]) # sum all white pixels in proposal region
            if(region_sum > 100000): # if more white pixels here than average
                south_delta = abs(r_counter-240)
                south_deltas.append(south_delta)
                region_sums.append(region_sum)
                r_region_counters.append(r_counter)
                c_region_counters.append(c_counter)
    # find southmost
    if(len(south_deltas) !=0):
        southmost_delta = np.amin(south_deltas)
        # find largest region within 50px of southmost point
        largest_south_index = 0
        for current_region in range(len(region_sums)):
            # print("south_deltas[current_region] - southmost_delta: ", south_deltas[current_region] - southmost_delta)
            if(region_sums[current_region] > region_sums[largest_south_index]):
                if( abs(south_deltas[current_region] - southmost_delta) < 50):
                    largest_south_index = current_region
        crop_coords = [c_region_counters[largest_south_index],c_region_counters[largest_south_index],r_region_counters[largest_south_index],r_region_counters[largest_south_index]]
        if(crop_coords[0]-15 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 15

        if(crop_coords[1]+100 >319): # how much extra right to do
            crop_coords[1] = 319
        else:
            crop_coords[1] += 100

        if(crop_coords[2] - 40 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 40

        if(crop_coords[3]+80 >239): # how much extra down to do
            crop_coords[3] = 239
        else:
            crop_coords[3] += 80
        cv2.imshow("image cropped", result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]]) # y1:y2,x1:x2
        # cv2.imshow("crop", image[0:10,0:100])
        # image[toptobottom,leftoright]
        # r_region_counters = 320 left to right
        # c_region_counters = 240 top to bottom
        cv2.imshow("image", cv2.rectangle(image, (c_region_counters[largest_south_index],r_region_counters[largest_south_index]), (c_region_counters[largest_south_index]+80,r_region_counters[largest_south_index]+60), (0,0,255), 2))
    else:
        cv2.imshow("image", image)
    if cv2.waitKey(5) == ord(" "):
        break    
cap.release()
