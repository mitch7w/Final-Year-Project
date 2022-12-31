import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

while True:
    for i in range(1,56):
        image_pathname = "/Users/mitch/Documents/University/Project/pinky_locator/gesture" + str(i) + ".jpg"
        image = cv2.imread(image_pathname)
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
        for r_counter in range(0,320,8):
            for c_counter in range(0,240,5):
                region_sum = np.sum(mask[r_counter:r_counter+100,c_counter:c_counter+120]) # sum all white pixels in proposal region
                if(region_sum > 100000): # if more white pixels here than average
                    south_delta = abs(r_counter-240)
                    south_deltas.append(south_delta)
                    region_sums.append(region_sum)
                    r_region_counters.append(r_counter)
                    c_region_counters.append(c_counter)
        # find southmost
        if(len(south_deltas) ==0):
            continue
        southmost_delta = np.amin(south_deltas)
        # find largest region within 50px of southmost point
        largest_south_index = 0
        for current_region in range(len(region_sums)):
            # print("south_deltas[current_region] - southmost_delta: ", south_deltas[current_region] - southmost_delta)
            if(region_sums[current_region] > region_sums[largest_south_index]):
                if( abs(south_deltas[current_region] - southmost_delta) < 50):
                    largest_south_index = current_region
        crop_coords = [r_region_counters[largest_south_index],r_region_counters[largest_south_index],c_region_counters[largest_south_index],c_region_counters[largest_south_index]]
        if(crop_coords[1]+100 >320):
            crop_coords[1] = 319
        else:
            crop_coords[1] += 100
        if(crop_coords[2]-40 <0):
            crop_coords[2] = 0
        else: 
            crop_coords[2] -= 40
        if(crop_coords[3]+150 >240):
            crop_coords[3] = 239
        else:
            crop_coords[3] += 150
        
        cv2.imshow("Small image", image[crop_coords[0]:crop_coords[1],crop_coords[2]:crop_coords[3]])

        if cv2.waitKey(500) == ord(" "):
            break
