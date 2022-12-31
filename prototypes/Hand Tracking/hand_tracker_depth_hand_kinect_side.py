import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import freenect
from PIL import ImageEnhance, Image
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

hand_depth = -1

while True:
    # read in from Kinect
    videoInput = freenect.sync_get_video()[0]
    depthInput = freenect.sync_get_depth()[0]
    # depthInput.shape = (480, 640)
    background_image = np.array(videoInput) 
    image = cv2.resize(background_image, (320,240), interpolation = cv2.INTER_AREA)
    cv2.imshow("Image", image)
    # increase saturation
    im_pil = Image.fromarray(image)
    converter = ImageEnhance.Color(im_pil)
    img2 = converter.enhance(0.5)
    # For reversing the operation:
    im_np = np.asarray(img2)
    image = cv2.cvtColor(im_np, cv2.COLOR_BGR2RGB)
    cv2.imshow("Enhancer", image)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
    # print(image_ycrcb[0][0])
    # skin1 = (0, 89, 136)
    # skin2 = (255, 147, 181)
    skin1 = (0, 89, 136)
    skin2 = (255, 200, 181)
    mask = cv2.inRange(image_ycrcb, skin1, skin2)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Color", result)
    cv2.imshow("Mask", mask) # 240 x 320 white or black image
    # determine if hand entering frame from top, left, bottom, right
    # mask is indexed with [height (top-bottom), width (left-right)]
    top_sum = np.sum(mask[0:10:,:])
    left_sum = np.sum(mask[:,0:10])
    bottom_sum = np.sum(mask[229:239,:])
    right_sum = np.sum(mask[:,309:319])
    edge_max_index = np.argmax([top_sum, left_sum, bottom_sum, right_sum]) # top, left, bottom, right axes entry of hand

    # find highest concentration of values in mask matrix closest to certain axes of image
    r_region_counters = []
    c_region_counters = []
    region_sums = []
    edge_deltas = []
    # find all regions with pixel count larger than 270 000 and find delta from desired axes
    for c_counter in range(0,320,5):
        for r_counter in range(0,240,5):
            region_sum = np.sum(mask[r_counter:r_counter+60,c_counter:c_counter+80]) # sum all white pixels in proposal region
            if(region_sum > 300000): # if more white pixels here than average
                if(edge_max_index==0): # hand coming in from top so find closest region to bottom
                    edge_delta = abs(r_counter-240)
                    # print("coming in from top")
                if(edge_max_index==1): # hand coming in from left so find closest region to right
                    edge_delta = abs(c_counter-320)
                    # print("coming in from left")
                if(edge_max_index==2): # hand coming in from bottom so find closest region to top
                    edge_delta = abs(r_counter)
                    # print("coming in from bottom")
                if(edge_max_index==3): # hand coming in from right so find closest region to left
                    edge_delta = abs(c_counter)
                    # print("coming in from right")
                edge_deltas.append(edge_delta)
                region_sums.append(region_sum)
                r_region_counters.append(r_counter)
                c_region_counters.append(c_counter)

    # find region closest to desired edge
    
    crop_coords = [0,80,0,60]
    # if there is actually a region that meets the required concentration of pixels
    if(len(edge_deltas) !=0):
        closest_delta_index = np.argmin(edge_deltas)
        crop_coords = [c_region_counters[closest_delta_index],c_region_counters[closest_delta_index],r_region_counters[closest_delta_index],r_region_counters[closest_delta_index]]
        depthFeed = frame_convert2.pretty_depth_cv(depthInput) # modified depthInput too
        depthFeedColoured = cv2.applyColorMap(depthFeed,cv2.COLORMAP_JET)
        # depthFeedColoured.shape = (480, 640, 3)
        
        # rectangle (x1,y1) (x2,y2) left to right, top to bottom
        # depthInput[0-480][0-640] # top to bottom, left to right
        # (480, 640)

    # widen extracted window to get the whole hand in the frame
    if(edge_max_index==0): # hand coming in from top
        if(crop_coords[0]-15 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 15

        if(crop_coords[1]+100 >319): # how much extra right to do
            crop_coords[1] = 319
        else:
            crop_coords[1] += 100

        if(crop_coords[2] - 80 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 80

        if(crop_coords[3]+80 >239): # how much extra down to do
            crop_coords[3] = 239
        else:
            crop_coords[3] += 80
        
    if(edge_max_index==1): # hand coming in from left
        if(crop_coords[0]-60 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 60

        if(crop_coords[1]+70 >319): # how much extra right to do
            crop_coords[1] = 319
        else:
            crop_coords[1] += 70

        if(crop_coords[2] - 40 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 40

        if(crop_coords[3]+80 >239): # how much extra down to do
            crop_coords[3] = 239
        else:
            crop_coords[3] += 80
        
    if(edge_max_index==2): # hand coming in from bottom
        if(crop_coords[0]-10 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 10
        if(crop_coords[1]+170 >319): # how much extra right to do
            crop_coords[1] = 319
        else:
            crop_coords[1] += 170
        if(crop_coords[2] - 40 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 40
        if(crop_coords[3]+120 >239): # how much extra down to do
            crop_coords[3] = 239
        else:
            crop_coords[3] += 120
        
    if(edge_max_index==3): # hand coming in from right
        if(crop_coords[0]-30 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 30
        if(crop_coords[1]+150 >319): # how much extra right to do
            crop_coords[1] = 319
        else:
            crop_coords[1] += 150
        if(crop_coords[2] - 40 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 40
        if(crop_coords[3]+120 >239): # how much extra down to do
            crop_coords[3] = 239
        else:
            crop_coords[3] += 120

    # display detected and extracted hand image
    if(len(edge_deltas) !=0):
        cv2.imshow("image cropped", result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]]) # y1:y2,x1:x2
        cv2.imshow("image rectangle", cv2.rectangle(image, (c_region_counters[closest_delta_index],r_region_counters[closest_delta_index]), (c_region_counters[closest_delta_index]+80,r_region_counters[closest_delta_index]+60), (0,0,255), 2))
        cv2.imshow("Background image", depthFeedColoured[int((crop_coords[2]/240)*480):int((crop_coords[3]/240)*480),int((crop_coords[0]/320)*640):int((crop_coords[1]/320)*640),:])
        cv2.imshow("Depth feed coloured", depthFeedColoured)
        hand_depth = np.min(depthInput[int((crop_coords[2]/240)*480):int((crop_coords[3]/240)*480),int((crop_coords[0]/320)*640):int((crop_coords[1]/320)*640)])
        print("hand_depth: ", hand_depth)
    if cv2.waitKey(5) == ord(" "):
        break    
