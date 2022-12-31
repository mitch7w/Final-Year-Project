import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageEnhance, Image
import cv2
sns.set_theme()
import freenect
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)

def handTracking(background_image,depthInput):
    image = cv2.resize(background_image, (80,60), interpolation = cv2.INTER_AREA)
    # increase saturation
    im_pil = Image.fromarray(image)
    converter = ImageEnhance.Color(im_pil)
    img2 = converter.enhance(0.5)
    # For reversing the operation:
    image = np.asarray(img2)
    # cv2.imshow("Enhancer", image)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB))
    # print(image_ycrcb[0][0])
    # skin1 = (0, 89, 136)
    # skin2 = (255, 147, 181)
    skin1 = (0, 100, 136)
    skin2 = (255, 144, 156)
    mask = cv2.inRange(image_ycrcb, skin1, skin2)
    result = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow("Color", result)
    cv2.imshow("Mask", mask) # 240 x 320 white or black image
    # determine if hand entering frame from top, left, bottom, right
    # mask is indexed with [height (top-bottom), width (left-right)]
    top_sum = np.sum(mask[0:2:,:])
    left_sum = np.sum(mask[:,0:2])
    bottom_sum = np.sum(mask[77:79,:])
    right_sum = np.sum(mask[:,57:59])
    edge_max_index = np.argmax([top_sum, left_sum, bottom_sum, right_sum]) # top, left, bottom, right axes entry of hand

    # find highest concentration of values in mask matrix closest to certain axes of image
    r_region_counters = []
    c_region_counters = []
    region_sums = []
    edge_deltas = []
    # find all regions with pixel count larger than 270 000 and find delta from desired axes
    for c_counter in range(0,80,2):
        for r_counter in range(0,60,2):
            region_sum = np.sum(mask[r_counter:r_counter+15,c_counter:c_counter+20]) # sum all white pixels in proposal region
            if(region_sum > 12000): # if more white pixels here than average
                if(edge_max_index==0): # hand coming in from top so find closest region to bottom
                    edge_delta = abs(r_counter-60)
                    # print("coming in from top")
                if(edge_max_index==1): # hand coming in from left so find closest region to right
                    edge_delta = abs(c_counter-80)
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
    
    crop_coords = [0,20,0,15]
    closest_delta_index = -1
    hand_present = False 
    # if there is actually a region that meets the required concentration of pixels
    if(len(edge_deltas) !=0):
        hand_present = True 
        closest_delta_index = np.argmin(edge_deltas)
        crop_coords = [c_region_counters[closest_delta_index],c_region_counters[closest_delta_index],r_region_counters[closest_delta_index],r_region_counters[closest_delta_index]]
        # depthFeed = frame_convert2.pretty_depth_cv(depthInput) # modified depthInput too
        # depthFeedColoured = cv2.applyColorMap(depthFeed,cv2.COLORMAP_JET)
        # depthFeedColoured.shape = (480, 640, 3)
        
        # rectangle (x1,y1) (x2,y2) left to right, top to bottom
        # depthInput[0-480][0-640] # top to bottom, left to right
        # (480, 640)

    # widen extracted window to get the whole hand in the frame
    if(edge_max_index==0): # hand coming in from top
        if(crop_coords[0]-5 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 5

        if(crop_coords[1]+25 >79): # how much extra right to do
            crop_coords[1] = 79
        else:
            crop_coords[1] += 25

        if(crop_coords[2] - 20 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 20

        if(crop_coords[3]+20 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 20
        
    if(edge_max_index==1): # hand coming in from left
        if(crop_coords[0]-15 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 15

        if(crop_coords[1]+18 >79): # how much extra right to do
            crop_coords[1] = 79
        else:
            crop_coords[1] += 18

        if(crop_coords[2] - 10 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 10

        if(crop_coords[3]+20 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 20
        
    if(edge_max_index==2): # hand coming in from bottom
        if(crop_coords[0]-3 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 3
        if(crop_coords[1]+43 >59): # how much extra right to do
            crop_coords[1] = 59
        else:
            crop_coords[1] += 43
        if(crop_coords[2] - 10 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 10
        if(crop_coords[3]+30 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 30
        
    if(edge_max_index==3): # hand coming in from right
        if(crop_coords[0]-7 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 7
        if(crop_coords[1]+37 >79): # how much extra right to do
            crop_coords[1] = 79
        else:
            crop_coords[1] += 37
        if(crop_coords[2] - 10 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 10
        if(crop_coords[3]+30 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 30

    # display detected and extracted hand image
    if(len(edge_deltas) !=0):
        cv2.imshow("image rectangle", cv2.rectangle(image, (c_region_counters[closest_delta_index],r_region_counters[closest_delta_index]), (c_region_counters[closest_delta_index]+20,r_region_counters[closest_delta_index]+15), (0,0,255), 2))
        # cv2.imshow("Background image", depthFeedColoured[int((crop_coords[2]/240)*480):int((crop_coords[3]/240)*480),int((crop_coords[0]/320)*640):int((crop_coords[1]/320)*640),:])
        # cv2.imshow("Depth feed coloured", depthFeedColoured)
        new_depth = np.double(np.min(depthInput[int((crop_coords[2]/60)*480):int((crop_coords[3]/60)*480),int((crop_coords[0]/80)*640):int((crop_coords[1]/80)*640)]))
    else:
        new_depth = -1
    hand_box_coords = (0,0)
    if(closest_delta_index != -1):
        hand_box_coords = (c_region_counters[closest_delta_index] , r_region_counters[closest_delta_index])
    extracted_hand = result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]]
    if(extracted_hand.shape != (40, 44, 3)):
        extracted_hand = cv2.resize(result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]], (44,40), interpolation = cv2.INTER_AREA)
    extracted_hand_greyscale = cv2.cvtColor(extracted_hand,cv2.COLOR_RGB2GRAY)
    cv2.imshow("extracted_hand_greyscale", extracted_hand_greyscale)
    # make sure all images are same size even if cropped on the side
    normalized_extracted_hand_greyscale = np.array(extracted_hand_greyscale)/255 # divide by 255 in order to normalize input
    if(len(normalized_extracted_hand_greyscale.shape) == 2): # if it's not 3 channels long put it in a 3D shape
        normalized_extracted_hand_greyscale = normalized_extracted_hand_greyscale.reshape((normalized_extracted_hand_greyscale.shape[0],normalized_extracted_hand_greyscale.shape[1],1))
    return hand_present, new_depth, hand_box_coords, np.array(normalized_extracted_hand_greyscale)

while True:
    # read in from Kinect
    videoInput = freenect.sync_get_video()[0]
    depthInput = freenect.sync_get_depth()[0]
    hand_status, hand_depth, hand_box_coords, extracted_hand = handTracking(np.array(videoInput), depthInput)
    # hand_box_coords going from 0 -> 60 on x and y axis
    # depthInput[0-480][0-640] # top to bottom, left to right (480, 640)
    print("hand_depth: ", hand_depth)
    depth_top_hand_box = np.min(depthInput[int(480*(hand_box_coords[0]/60))-10][hand_box_coords[1]:hand_box_coords[1]+10])
    print("depth_top_hand_box: ", depth_top_hand_box)