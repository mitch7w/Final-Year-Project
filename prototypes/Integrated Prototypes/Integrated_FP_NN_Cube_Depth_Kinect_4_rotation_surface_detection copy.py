# import statements
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import freenect
from PIL import ImageEnhance, Image
import pygame
import time
import scipy.signal
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import importlib.util
from numpy.random import default_rng
import numba as nb
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)
np.set_printoptions(threshold=sys.maxsize) # makes arrays print out in full for debugging 

rng = default_rng(777) # set random seed for random number generator in NN weights setting

############### global variables needed for OpenGL augmented reality rendering ###############

previous_depth = np.double(-99) # previous depth value of hand
hand_x_coord = 0
hand_y_coord = 0
# cube x and cube y coords = coords for bottom left of front face of cube
cube_x_coord = 0.5 # 0-11 left to right of screen 
cube_y_coord = 0.5 # 0-8 top to bottom of screen
cube_z_coord = 0
cube_scale = 1
x_rotation_degrees = 0
y_rotation_degrees = 0
z_rotation_degrees = 0
rotation_history = []

# create the cube with the defined vertices and edges
def Cube():
    vertices= (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )
    edges = (
        (0,1),
        (0,3),
        (0,4),
        (2,1),
        (2,3),
        (2,7),
        (6,3),
        (6,4),
        (6,7),
        (5,1),
        (5,4),
        (5,7)
        )
    colors = (
    (1,0,0),
    (1,0,1),
    (1,0,0),
    (1,1,0),
    (1,1,1),
    (0,1,1),
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,0,0),
    (1,1,1),
    (0,1,1),
    )
    surfaces = (
    (0,1,2,3),
    (3,2,7,6),
    (6,7,5,4),
    (4,5,1,0),
    (1,5,7,2),
    (4,0,3,6)
    )
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x+=1
            glColor3fv(colors[x]) # color each surface with a slightly different color
            glVertex3fv(vertices[vertex])
    glEnd()
    # glBegin denotes start of special OpenGL commands to follow
    glBegin(GL_LINES) 
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex]) # render each vertex
    glEnd() # end of special OpenGL commands

# get new hand position and move/scale cube
def move_and_scale_cube(startup_depthInput, startup_detected_surface, normalized_hand_x, normalized_hand_y, hand_depth,collisions):
    global hand_x_coord, hand_y_coord, cube_x_coord, cube_y_coord, previous_depth, cube_scale
    
    # change hand coords to be in relation to OpenGL coords
    hand_x_coord = (normalized_hand_x*11.0)
    hand_y_coord = (normalized_hand_y*8.0)
    
    # check if hand close to cube and enable movement
    if(hand_depth != -1 or hand_depth != 2047):# hand been detected and is not noisey depth data
        # hand distances from cube
        x_delta = hand_x_coord - cube_x_coord
        y_delta = hand_y_coord - cube_y_coord
        # if hand close to cube, move cube to hand position and scale
        if(abs(x_delta) < 2.0 and abs(y_delta) <2.0): 
            
            # perfom surface collision avoidance
            surf_time = time.time()
            collision_status = surface_collision(startup_depthInput, startup_detected_surface, hand_x_coord,hand_y_coord, hand_depth)
            # print("Surf time: ", time.time()-surf_time)
            # collision_status = "nocollision"
            # print("Surface detection: ", collision_status)
            if(collision_status=="nocollision"):
                # now do object detection
                # sides_to_check_for_collisions = [0,0,0,0] # top left bottom right
                # if(x_delta > 0): sides_to_check_for_collisions[3] = 1 # need to check right side as hand on right of cube
                # if(x_delta < 0): sides_to_check_for_collisions[1] = 1 # need to check left side as hand on left of cube
                # if(y_delta < 0): sides_to_check_for_collisions[0] = 1 # need to check top side as hand on top of cube
                # if(y_delta > 0): sides_to_check_for_collisions[2] = 1 # need to check bottom side as hand on bottom of cube
                allowed_to_move = True
                # for side in range(len(sides_to_check_for_collisions)):
                #     if(sides_to_check_for_collisions[side] ==1):
                #         if(collisions[side] ==1):
                #             allowed_to_move = False
                # print("Allowed to move: ", allowed_to_move)
                if(allowed_to_move):

                    cube_x_coord = hand_x_coord
                    cube_y_coord = hand_y_coord
                    und_rotate_time = time.time()
                    # undo all rotations
                    current_rotation_history = rotation_history.copy()
                    current_rotation_history.reverse() # do the rotations in reverse order
                    for each_rotation in current_rotation_history:
                        if(each_rotation[0] == "x"):
                            glRotatef(-each_rotation[1],1,0,0)
                        if(each_rotation[0] == "y"):
                            glRotatef(-each_rotation[1],0,1,0)
                        if(each_rotation[0] == "z"):
                            glRotatef(-each_rotation[1],0,0,1)
                    
                    glTranslatef(-1.0,-1.0,-1.0)
                    glTranslatef(x_delta,-y_delta, 0) # actually move cube to hand position
                    glTranslatef(1.0,1.0,1.0)
                    
                    # redo all rotations
                    for each_rotation in rotation_history:
                        if(each_rotation[0] == "x"):
                            glRotatef(each_rotation[1],1,0,0)
                        if(each_rotation[0] == "y"):
                            glRotatef(each_rotation[1],0,1,0)
                        if(each_rotation[0] == "z"):
                            glRotatef(each_rotation[1],0,0,1)

                    print("Undo rotation time: ", time.time()-und_rotate_time)       
                    # scale cube
                    if(previous_depth == -99): # first time detecting a hand
                        previous_depth = hand_depth
                    else:
                        if(hand_depth>=300 and hand_depth <= 700): # if hand depth in correct range to be detected
                            depth_delta = (previous_depth-hand_depth)/800
                            cube_scale = (1+depth_delta)
                            previous_depth = hand_depth
                            # scale cube to correct z coord
                            glTranslatef(-1.0,-1.0,-1.0)
                            glScale(cube_scale,cube_scale, cube_scale) 
                            glTranslatef(1.0,1.0,1.0)

# Find hand location, hand depth and extracted image of hand
def handTracking(background_image,depthInput):
    image = cv2.resize(background_image, (80,60), interpolation = cv2.INTER_AREA) # resize image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # increase saturation
    im_pil = Image.fromarray(image)
    converter = ImageEnhance.Color(im_pil)
    img2 = converter.enhance(0.5)
    image = np.asarray(img2)
    # cv2.imshow("Enhancer", image)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)) # convert to Ycbcr colour space
    # skin hue range
    skin1 = (91, 94, 134)
    skin2 = (255, 131, 160)
    mask = cv2.inRange(image_ycrcb, skin1, skin2) # only get pixels in skin range
    result = cv2.bitwise_and(image, image, mask=mask) # create binary image
    cv2.imshow("Color", result)
    cv2.imshow("Mask", mask) # 240 x 320 white or black image
    # determine if hand entering frame from top, left, bottom, right
    # mask is indexed with [height (top-bottom), width (left-right)]
    top_sum = np.sum(mask[0:2:,:])
    left_sum = np.sum(mask[:,0:2])
    bottom_sum = np.sum(mask[77:79,:])
    right_sum = np.sum(mask[:,57:59])
    edge_max_index = np.argmax([top_sum, left_sum, bottom_sum, right_sum])

    def find_hand_regions_get_crop_coords(mask):
        # find highest concentration of values in mask matrix closest to certain axes of image (find hand)
        r_region_counters = []
        c_region_counters = []
        region_sums = []
        edge_deltas = []
        # find all regions with pixel count larger than threshold and find delta from desired axes
        for c_counter in range(0,80,5):
            for r_counter in range(0,60,5):
                region_sum = np.sum(mask[r_counter:r_counter+15,c_counter:c_counter+20]) # sum all white pixels in proposal region
                if(region_sum > 12000): # if more white pixels here than threshold
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
                    # append region and its parameters to arrays
                    edge_deltas.append(edge_delta)
                    region_sums.append(region_sum)
                    r_region_counters.append(r_counter)
                    c_region_counters.append(c_counter)

        # find region closest to desired edge
        crop_coords = [0,20,0,15] # coordinates to crop image with
        closest_delta_index = -1
        hand_present = False 
        if(len(edge_deltas) !=0): # if there is actually a region that meets the required concentration of pixels
            hand_present = True 
            closest_delta_index = np.argmin(edge_deltas) # region closest to desired edge
            # print("c_counter: ", c_region_counters[closest_delta_index], ", r_region: ", r_region_counters[closest_delta_index])
            crop_coords = [c_region_counters[closest_delta_index],c_region_counters[closest_delta_index],r_region_counters[closest_delta_index],r_region_counters[closest_delta_index]] # modify crop coords to desired region

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
        return edge_deltas , c_region_counters, r_region_counters, closest_delta_index, hand_present, crop_coords

    edge_deltas , c_region_counters, r_region_counters, closest_delta_index, hand_present, crop_coords = find_hand_regions_get_crop_coords(mask)

    # display detected and extracted hand image
    if(len(edge_deltas) !=0):
        cv2.imshow("image rectangle", cv2.rectangle(image, (c_region_counters[closest_delta_index],r_region_counters[closest_delta_index]), (c_region_counters[closest_delta_index]+20,r_region_counters[closest_delta_index]+15), (0,0,255), 2))
        # cv2.imshow("Background image", depthFeedColoured[int((crop_coords[2]/240)*480):int((crop_coords[3]/240)*480),int((crop_coords[0]/320)*640):int((crop_coords[1]/320)*640),:])
        # cv2.imshow("Depth feed coloured", depthFeedColoured)
        new_depth = np.double(np.min(depthInput[int((crop_coords[2]/60)*480):int((crop_coords[3]/60)*480),int((crop_coords[0]/80)*640):int((crop_coords[1]/80)*640)])) # get closest part of hand in hand region
    else:
        new_depth = -1
    hand_box_coords = (0,0)
    # print("result.shape: ", len(result[0]))
    if(closest_delta_index != -1):
        hand_box_coords = (c_region_counters[closest_delta_index] , r_region_counters[closest_delta_index]) # get coords of hand region
    extracted_hand = result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]] # extract part of image that contains hand
    if(extracted_hand.shape != (40, 44, 3)):
        extracted_hand = cv2.resize(result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]], (44,40), interpolation = cv2.INTER_AREA) # resize image for consistent NN inference
    # extracted_hand_greyscale = cv2.cvtColor(extracted_hand,cv2.COLOR_RGB2GRAY) # convert image to greyscale
    # convert hand box coords to Kinect window scale
    hand_x = int((hand_box_coords[0]/80)*640)
    hand_y = int((hand_box_coords[1]/60)*480)
    cv2.imshow("extracted_hand", extracted_hand)
    normalized_extracted_hand = np.array(extracted_hand)/255 # divide by 255 in order to normalize input
    if(len(normalized_extracted_hand.shape) == 2): # if image not 3 channels long put it in a 3D shape (for NN multidimensional input sake)
        normalized_extracted_hand = normalized_extracted_hand.reshape((normalized_extracted_hand.shape[0],normalized_extracted_hand.shape[1],1))
    return hand_present, new_depth, (hand_x,hand_y), np.array(normalized_extracted_hand), edge_max_index

# extract depth values around hand region for object detection/collision avoidance
def depth_around_object(depthInput, hand_bounding_box, side_entered_from):
    # middle of hand region coords
    middle_of_hand_x = hand_bounding_box[0]+40
    middle_of_hand_y = hand_bounding_box[1]+30
    if(middle_of_hand_x > 539): middle_of_hand_x = 539
    if(middle_of_hand_y > 409): middle_of_hand_y = 409
    # get depth values for rows and cols just to top, bottom, left and right of bounding box around hand
    
    top_hbox_coord = hand_bounding_box[1]+10 # +10 because of crop_coords artificially widening extracted image
    if(top_hbox_coord<0): top_hbox_coord=0
    if(top_hbox_coord>479): top_hbox_coord=479
    # display box
    # depthInput[top_hbox_coord:top_hbox_coord+10,middle_of_hand_x+60:middle_of_hand_x+80] = np.zeros(depthInput[top_hbox_coord:top_hbox_coord+10,middle_of_hand_x+60:middle_of_hand_x+80].shape)
    depth_top_hand_box = np.min(depthInput[top_hbox_coord:top_hbox_coord+10,middle_of_hand_x+60:middle_of_hand_x+80])

    bottom_hbox_coord = middle_of_hand_y+130
    if(bottom_hbox_coord>469): bottom_hbox_coord = 469
    if(bottom_hbox_coord<0): bottom_hbox_coord=0
        # display box
        # depthInput[bottom_hbox_coord:bottom_hbox_coord+10,middle_of_hand_x+80:middle_of_hand_x+100] = np.zeros(depthInput[bottom_hbox_coord:bottom_hbox_coord+10,middle_of_hand_x+80:middle_of_hand_x+100].shape)
    depth_bottom_hand_box = np.min(depthInput[bottom_hbox_coord:bottom_hbox_coord+10,middle_of_hand_x+80:middle_of_hand_x+100])
    
    left_hbox_coord = hand_bounding_box[0] + 50
    if(left_hbox_coord < 0): left_hbox_coord = 0
    if(left_hbox_coord > 639): left_hbox_coord = 639
        # display box
        # depthInput[middle_of_hand_y+30:middle_of_hand_y+50,left_hbox_coord:left_hbox_coord+10] = np.zeros(depthInput[middle_of_hand_y+30:middle_of_hand_y+50,left_hbox_coord:left_hbox_coord+10].shape)
    depth_left_hand_box = np.min(depthInput[middle_of_hand_y+30:middle_of_hand_y+50,left_hbox_coord:left_hbox_coord+10])

    right_hbox_coord = middle_of_hand_x + 250
    if(right_hbox_coord > 629): right_hbox_coord = 629
    if(right_hbox_coord < 0): right_hbox_coord = 0
        # display box
        # depthInput[middle_of_hand_y+50:middle_of_hand_y+70,right_hbox_coord:right_hbox_coord+10] = np.zeros(depthInput[middle_of_hand_y+50:middle_of_hand_y+70,right_hbox_coord:right_hbox_coord+10].shape)
    depth_right_hand_box = np.min(depthInput[middle_of_hand_y+50:middle_of_hand_y+70,right_hbox_coord:right_hbox_coord+10])

    depth_boxes = [depth_top_hand_box, depth_left_hand_box, depth_bottom_hand_box, depth_right_hand_box]
    # side_entered_from = 0,1,2,3 = top, left, bottom, right
    depth_boxes[side_entered_from] = -1 # don't return a depth for where we know the arm enters - want object not arm depth
    return depth_boxes , depthInput

############## Different layers available for use in neural network ##############

class Input:
    output_neurons = []
  
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def get_size(self):
        return len(self.output_neurons)

    def get_output(self):
        return self.output_neurons

    def forward_propagation(self,new_neurons):
        self.output_neurons = new_neurons
        # writeToFile("input_layer: " + str(self.output_neurons))

class Convolution:
    output_neurons = []
    filters = [] # filter values in shape of (rows, cols, channels, numFilters). E.g. 3x3 greyscale with 4 filters = (3, 3, 1, 4)
    bias = [] # 1 bias for each filter
    delta = [] # delta values to backpropagate error
    
    # initialise values
    def __init__(self, number_filters, kernel_size, activation_function):
        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        # initialize filters with random values from -1 to 1
        self.filters = rng.random(number_filters*kernel_size[0]*kernel_size[1])
        for f in range(len(self.filters)):
            self.filters[f] = (self.filters[f] *2) -1  # generate new filter value -1 to 1
        self.filters=self.filters.reshape(number_filters,kernel_size[0],kernel_size[1])

    def get_size(self):
            return self.number_neurons

    def activation(self, x, activation_function):
        if(activation_function=="relu"):
            return x * (x > 0)
        if(activation_function=="sigmoid"):
            return 1/(1+np.e**(-x))

    def activation_derivative(self, x, activation_function):
        if(activation_function=="relu"):
            if(x>0):
                return 1
            else:
                return 0
        if(activation_function=="sigmoid"):
            z= 1/(1+np.e**(-x))
            return z * (1-z)

    # activate each value in a matrix with chosen activation function
    def activate_matrix(self,matrix,bias):
        matrix = matrix + bias # add the bias to each value of a matrix
        return self.activation( matrix, self.activation_function)

    # convolution operation for an input image and filter
    def convolve(self,input, filter):
        return scipy.signal.correlate2d(input, filter, mode='valid')

    # slide filters over input image and get convolved output
    def forward_propagation(self, previous_layer_output):
        conv_start_time = time.time()
        # filter values in shape of (rows, cols, channels, numFilters). E.g. 3x3 greyscale with 4 filters = (3, 3, 1, 4)
        # bias = 1 bias for each filter
        output_size = ( (previous_layer_output.shape[0]-self.kernel_size[0]+1) , (previous_layer_output.shape[1]-self.kernel_size[1]+1), self.number_filters ) # define output size based on kernel and stride size
        self.output_neurons = np.zeros(output_size)
        # convolve each RGB channel with the relevant filter channel
        for num_filter in range(self.number_filters): # for each filter
            convolution_outputs = np.zeros( (output_size[0],output_size[1]) )
            for num_channel in range(previous_layer_output.shape[2]):
                # convolve each filter channel with each input image channel and sum all the channel outputs together
                convolution_outputs = convolution_outputs + self.convolve(previous_layer_output[:,:,num_channel], self.filters[:,:,num_channel,num_filter])
            # set each part of convolved output matrix after adding bias and activating
            self.output_neurons[:,:,num_filter] = self.activate_matrix(convolution_outputs, self.bias[num_filter])
        # print(f"Conv runtime: {(time.time()  - conv_start_time):.2f} seconds")
        # print("Shape at output of conv. layer 1: ", self.output_neurons.shape)
        # writeToFile("convolution_output: " + str(self.output_neurons))
        # writeToFile("convolution_output shape: " + str(self.output_neurons.shape))
        
    # propagate error backwards through convolution layer
    def backpropagation(self, next_layer_delta, xy_max_pairs, input_data,learning_rate):
        convolution_gradient = np.zeros(self.output_neurons.shape[0]*self.output_neurons.shape[1]*self.number_filters).reshape((self.output_neurons.shape[0],self.output_neurons.shape[1],self.number_filters))
        # for every max-pooled value set its gradient based on next layer delta
        next_layer_delta = np.array(next_layer_delta).flatten()
        next_layer_delta_counter = 0
        for num_filter in range(self.number_filters):
            for x in range(len(xy_max_pairs)):
                for y in range(len(xy_max_pairs[x])):
                    convolution_gradient[int(xy_max_pairs[x][y][num_filter][0])][int(xy_max_pairs[x][y][num_filter][1])][num_filter] = next_layer_delta[next_layer_delta_counter]
                    next_layer_delta_counter +=1
        # find filter gradients based on delta values and input and update filter values for each R,G,B filter
        filter_gradients = np.empty(self.filters.shape)
        for num_filter in range(self.number_filters):
            channel_gradients = np.empty((self.filters.shape[1],self.filters.shape[2],self.filters.shape[3]))
            for channel_num in range(len(input_data[0][0])):
                channel_gradients[:,:,channel_num] = self.convolve(input_data[:,:,channel_num],convolution_gradient[:,:,num_filter])
            filter_gradients[num_filter] = channel_gradients
        # update each filter value
        for num_filter in range(len(self.filters)):
            for channel_num in range(len(input_data[0][0])):
                for x in range(len(self.filters[num_filter])):
                    for y in range(len(self.filters[num_filter][x])):
                        self.filters[num_filter][x][y][channel_num] -= learning_rate * filter_gradients[num_filter][x][y][channel_num]
        # update input matrix X with full convolvution of 180deg rotated filter and gradient
        # rotate each element of filter about vertical then horizontal axis
        rotatedFilters = self.filters.copy()
        rotatedFilters = rotatedFilters.flatten()
        rotatedFilters = np.flip(rotatedFilters)
        rotatedFilters = rotatedFilters.reshape(self.filters.shape)
        self.delta = []
        for num_filter in range(len(rotatedFilters)):
            channelsDelta = []
            for channel_num in range(len(rotatedFilters[num_filter][0][0])):
                channelsDelta.append(self.fullConvolution(rotatedFilters[num_filter][:,:,channel_num],convolution_gradient[:,:,num_filter]))
            self.delta.append(channelsDelta)
        # writeToFile("Filters: " + str(self.filters))

    def get_output(self):
        return self.output_neurons

    def get_delta(self):
        return self.delta

class Maxpooling:
    output_neurons = []
    delta = []
    xy_max_pairs = []

    def __init__(self, pool_size, stride_size):
        self.pool_size = pool_size
        self.stride_size = stride_size

    def get_output(self):
        return self.output_neurons

    # get max pooled value of areas according to pool and stride size
    def forward_propagation(self, previous_layer_output):
        # some operations commented out to save processing time when not doing backprop.
        maxpool_start_time = time.time()
        # set size of variables based on stride, pool and input size
        # self.xy_max_pairs = np.zeros((int((((previous_layer_output.shape[0]-self.pool_size[0])/self.stride_size)+1)) , int((((previous_layer_output.shape[1]-self.pool_size[1])/self.stride_size)+1)) , previous_layer_output.shape[2], 2))
        self.output_neurons = np.zeros( ( int( (previous_layer_output.shape[0]-self.pool_size[0]) /self.stride_size ) +1 , int( (previous_layer_output.shape[1]-self.pool_size[1]) /self.stride_size ) +1 , previous_layer_output.shape[2]) ) # new dimensions after pooling
        max_pooled_values = []
        # inputX and inputY are counters over input layer
        inputX = 0
        inputY = 0
        # iterate over the entire input matrix in pools of pool_size[0] x pool_size[1]
        while(inputX+self.pool_size[0] < previous_layer_output.shape[0]+1):
            while(inputY+self.pool_size[1] < previous_layer_output.shape[1]+1):
                for filter_num in range(previous_layer_output.shape[2]):
                    # filterX and filterY are the x and y positions going across the length of the pooling matrix
                    # iterate over the whole pooling matrix
                    maxValue = np.max(previous_layer_output[inputX:inputX+self.pool_size[0],inputY:inputY+self.pool_size[1],filter_num]) # find max value in pool
                    max_pooled_values.append(maxValue)
                    # self.output_neurons[inputX][inputY][filter_num] = maxValue # set each element of pool matrix = max value
                    # self.xy_max_pairs[inputX][inputY][filter_num] = (x_max,y_max) # store where max element was so can find it easily in backpropogation
                inputY += self.stride_size
            inputX += self.stride_size
            inputY = 0
        self.output_neurons = np.array(max_pooled_values).reshape(( int( (previous_layer_output.shape[0]-self.pool_size[0]) /self.stride_size ) +1 , int( (previous_layer_output.shape[1]-self.pool_size[1]) /self.stride_size ) +1 , previous_layer_output.shape[2]))
        # print("Shape at output of Maxpooling layer: ", self.output_neurons.shape)
        # print(f"Maxpool runtime: {(time.time()  - maxpool_start_time):.2f} seconds")
        # writeToFile("Maxpooling layer: " + str(self.output_neurons))
        # writeToFile("Maxpooling layer size : " + str(self.output_neurons.shape))

    def backpropagation(self,next_layer_delta):
        self.delta = next_layer_delta # just pass previous error backwards since this layer has no learned parameters

    def get_delta(self):
        return self.delta

    def get_xy_max_pairs(self):
        return self.xy_max_pairs

class Flatten:
    output_neurons = []
    flatten_delta = []

    def get_size(self):
        return len(self.output_neurons)

    def get_output(self):
        return self.output_neurons

    def forward_propagation(self, previous_layer_output):
        flatten_time = time.time()
        self.output_neurons = np.array(previous_layer_output).flatten() # turn multi-dimensional array into 1D array
        # writeToFile("Flatten layer: " + str(self.output_neurons))

        # print(f"Flatten runtime: {(time.time()  - flatten_time):.2f} seconds")

    def backpropagation(self,next_layer_delta,next_layer_weights,activation_derivative,activation_function):
        # calculate errors for input layer
        input_errors = [0] * len(self.output_neurons)
        for i in range(len(next_layer_weights)): # weights_1[i] = each hidden neuron
            for j in range(len(next_layer_weights[i])): # weights_1[i][j] = hidden neuron's connection to each input layer neuron
                input_errors[j] += next_layer_weights[i][j] * next_layer_delta[i]
        self.flatten_delta = []
        for i in range(len(input_errors)):
            self.flatten_delta.append(input_errors[i] * activation_derivative(self.output_neurons[i],activation_function))
    
    def get_delta(self):
        return self.flatten_delta

class Hidden:
    output_neurons = []
    hidden_delta = []

    def __init__(self, number_neurons, activation_function):
        self.number_neurons = number_neurons
        self.activation_function = activation_function
    
    def get_size(self):
        return self.number_neurons

    # initialize weights and biases with the He method
    def initialize_He_weights(self,previous_layer_output_size):
        # set weights and biases to random values
        self.weights = rng.random(previous_layer_output_size*self.number_neurons)
        self.bias = rng.random(self.number_neurons)
        standard_deviation = np.sqrt(2/previous_layer_output_size) # find std based on input size
        # modify weights and biases with the std
        for w in range(len(self.weights)):
            self.weights[w] = self.weights[w] * standard_deviation
        self.weights=self.weights.reshape(self.number_neurons,previous_layer_output_size)
        for b in range(len(self.bias)):
            self.bias[b] =self.bias[b] * standard_deviation

    def get_output(self):
        return self.output_neurons

    def activation(self, x, activation_function):
        if(activation_function=="relu"):
            return x * (x > 0)
        if(activation_function=="sigmoid"):
            return 1/(1+np.e**(-x))

    def activation_derivative(self, x, activation_function):
        if(activation_function=="relu"):
            if(x>0):
                return 1
            else:
                return 0
        if(activation_function=="sigmoid"):
            z= 1/(1+np.e**(-x))
            return z * (1-z)
    
    def forward_propagation(self, previous_layer_output):
        hidden_time = time.time()
        #weights are (input_connections, num_hidden_neurons)
        self.output_neurons=[]
        for neuron_number in range(self.number_neurons):
            # update hidden layer with dot product of inputs * weights all run through the activation function
            activationInput = np.dot(previous_layer_output , self.weights[:,neuron_number])+ self.bias[neuron_number]
            self.output_neurons.append(self.activation(activationInput, self.activation_function))
        # print(f"Hidden runtime: {(time.time()  - hidden_time):.2f} seconds")
        # writeToFile("Hidden layer: " + str(self.output_neurons))

    def backpropagation(self,learning_rate,next_layer_delta,previous_layer_output,next_layer_weights):
        # calculate errors for hidden layer
        hidden_errors = [0] * self.number_neurons
        for i in range(len(next_layer_weights)): # self.weights[i] = each hidden neuron's weight array
            for j in range(len(next_layer_weights[i])): # self.weights[i][j] = hidden neuron's connection to each previous layer jth neuron
                # for each neuron in hidden layer, add to error the weight*hidden delta
                hidden_errors[j] += next_layer_weights[i][j] * next_layer_delta[i]
        #calculate delta for hidden layer
        self.hidden_delta = []
        for i in range(len(hidden_errors)):
            self.hidden_delta.append(hidden_errors[i] * self.activation_derivative(self.output_neurons[i],self.activation_function))
        # update weights
        for i in range(len(self.weights)): # self.weights[i] = each hidden neuron
            for j in range(len(self.weights[i])): # self.weights[i][j] = hidden neuron's connection to each previous layer jth neuron
                self.weights[i][j] -= learning_rate * self.hidden_delta[i] * previous_layer_output[j]
            # update biases
            self.bias[i] -= learning_rate * self.hidden_delta[i]
        # writeToFile("hidden_layer_weights: " + str(self.weights))

    def get_weights(self):
        return self.weights

    def get_delta(self):
        return self.hidden_delta
    
    def get_biases(self):
        return self.bias

class Output:
    # Only difference from hidden is how backpropagation is conducted - now with desired training output
    output_neurons = []
    output_delta = []

    def __init__(self, number_neurons, activation_function):
        self.number_neurons = number_neurons
        self.activation_function = activation_function
    
    def get_size(self):
        return self.number_neurons

    # initialize weights and biases with the He method
    def initialize_He_weights(self,previous_layer_output_size):
        # set weights and biases to random values
        self.weights = rng.random(previous_layer_output_size*self.number_neurons)
        self.bias = rng.random(self.number_neurons)
        standard_deviation = np.sqrt(2/previous_layer_output_size) # find std based on input size
        # modify weights and biases with the std
        for w in range(len(self.weights)):
            self.weights[w] = self.weights[w] * standard_deviation
        self.weights=self.weights.reshape(self.number_neurons,previous_layer_output_size)
        for b in range(len(self.bias)):
            self.bias[b] =self.bias[b] * standard_deviation

    def get_output(self):
        return self.output_neurons

    def activation(self, output_array, activation_function):
        if(activation_function=="softmax"):
            exponents_output_array = np.exp(output_array)
            return exponents_output_array / np.sum(exponents_output_array)

    def activation_derivative(self, x, activation_function):
        if(activation_function=="relu"):
            return x * (x > 0)
        if(activation_function=="sigmoid"):
            z= 1/(1+np.e**(-x))
            return z * (1-z)
    
    def forward_propagation(self, previous_layer_output):
        output_time = time.time()
        self.output_neurons=[]
        for neuron_number in range(self.number_neurons):
            # update hidden layer with dot product of inputs * weights all run through the activation function
            activationInput = np.dot(previous_layer_output , self.weights[:,neuron_number])+ self.bias[neuron_number]
            self.output_neurons.append(activationInput)
        self.output_neurons = self.activation(self.output_neurons, self.activation_function) # activate output
        # print(f"Output runtime: {(time.time()  - output_time):.2f} seconds")
        # writeToFile("Output layer: " + str(self.output_neurons))

    def backpropagation(self,desired_output,learning_rate,previous_layer_output):
        # calculate errors for output layer
        output_errors = np.array(self.output_neurons) - np.array(desired_output)
        # calculate delta of output layer - how much they must change
        self.output_delta = []
        for i in range(len(output_errors)):
            self.output_delta.append(output_errors[i] * self.activation_derivative(self.output_neurons[i],self.activation_function))
        # update weights
        for i in range(len(self.weights)): # self.weights[i] = each output neuron
            for j in range(len(self.weights[i])): # self.weights[i][j] = output neuron's connection to each previous layer neuron
                self.weights[i][j] -= learning_rate * self.output_delta[i] * previous_layer_output[j]
            # update output biases
            self.bias[i] -= learning_rate * self.output_delta[i]
        # writeToFile("output_layer_weights: " + str(self.weights))
            
    def get_delta(self):
        return self.output_delta

    def get_weights(self):
        return self.weights
    
    def get_biases(self):
        return self.bias

# Encapsulating class for entire neural network
class Network:
    epochErrors = []
    networkInitialized = False # once forward prop. is run once and all paramaters initialised, this is True

    def __init__(self, layers):
        self.layers = layers

    # perform forward propogation of whole network based on new input data
    def forward_propagation(self, new_input_data):
        for i in range(len(self.layers)):
            if(i==0):
                self.layers[i].forward_propagation(new_input_data) # pass in the new input data
            else:
                # Initialize weights with He method if first time through network
                if(self.networkInitialized == False):
                    if(isinstance(self.layers[i], Hidden) or isinstance(self.layers[i], Output)):
                        self.layers[i].initialize_He_weights(self.layers[i-1].get_size())
                # propagate forwards with previous layer's output as input to next layer
                self.layers[i].forward_propagation(self.layers[i-1].get_output())
        self.networkInitialized = True
        
    # run a new input image through forward prop. and get a prediction
    def predict(self, new_input_data):
        total_start_time = time.time()
        self.forward_propagation(new_input_data)
        # print(f"Total runtime: {(time.time()  - total_start_time):.2f} seconds")
        return self.layers[len(self.layers)-1].get_output()

    # backprop. the error of a desired output through the whole network and update all parameters
    def backpropagation(self, input_data, desired_output,learning_rate):
        for i in range(len(self.layers)-1,-1,-1): # loop backwards through layers
            # pass in different paramaters based on type of layer
            if(isinstance(self.layers[i], Output)):
                self.layers[i].backpropagation(desired_output,learning_rate,self.layers[i-1].get_output()) # last layer
            elif(isinstance(self.layers[i], Hidden)):
                self.layers[i].backpropagation(learning_rate,self.layers[i+1].get_delta(),self.layers[i-1].get_output(),self.layers[i+1].get_weights())
            elif(isinstance(self.layers[i], Flatten)):
                self.layers[i].backpropagation(self.layers[i+1].get_delta(),self.layers[i+1].get_weights(),self.layers[i+1].activation_derivative,self.layers[i+1].activation_function)
            elif(isinstance(self.layers[i], Maxpooling)):
                self.layers[i].backpropagation(self.layers[i+1].get_delta())
            elif(isinstance(self.layers[i], Convolution)):
                self.layers[i].backpropagation(self.layers[i+1].get_delta(),self.layers[i+1].get_xy_max_pairs(),self.layers[i-1].get_output(),learning_rate)
            elif(isinstance(self.layers[i], Input)):# last layer
                pass # no learnable paramaters at input so can finish backpropagation process now.
    
    # take in a dataset and train the network parameters based on the data
    def train(self, training_data_input, training_data_output, learning_rate, num_training_epochs):
        # writeToFile("learning_rate: " + str(learning_rate))
        # writeToFile("num_training_epochs: " + str(num_training_epochs))
        # the networks sees all the data num_training_epochs times
        for epoch in range(num_training_epochs):
            epoch_start_time = time.time()
            total_error_at_output = 0
            # for each set of training data calculate the total error
            length_training_data = len(training_data_input)
            for training_data_num in range(length_training_data):
                self.forward_propagation(training_data_input[training_data_num])
                total_error_at_output += sum( (np.array(training_data_output[training_data_num]) -np.array(self.layers[len(self.layers)-1].get_output()))**2)
                self.back_propagation(training_data_input[training_data_num], training_data_output[training_data_num],learning_rate)        
                currentProgressString = str(training_data_num) + "/", str(length_training_data)
                print(training_data_num, "/", length_training_data, end='\r')
            epoch_end_time = time.time()
            # print current training progress
            print("Epoch: ", epoch, end=", ")
            print("Learning rate: ", learning_rate, end=", ")
            print("Error: ", total_error_at_output, end=", ")
            print(f"Runtime: {(epoch_end_time - epoch_start_time):.2f} seconds")
            self.epochErrors.append(total_error_at_output)
            # save weights for later use
            if(total_error_at_output < 1):
                np.save('Trained Nets/NN_gesture_layers_hidden_weights', model.layers[2].get_weights())
                np.save('Trained Nets/NN_gesture_layers_output_weights', model.layers[3].get_weights())
                np.save('Trained Nets/NN_gesture_layers_hidden_bias', model.layers[2].get_biases())
                np.save('Trained Nets/NN_gesture_layers_output_bias', model.layers[3].get_biases())
        # plot loss curve
        plt.plot(range(len(self.epochErrors)), self.epochErrors)
        plt.xlabel("Num Epochs")
        plt.ylabel("Training Square Loss")
        plt.show()

        # Overfitting dataset checks
        for i in range(len(training_data_input)):
            prediction = self.predict(training_data_input[i])
            print("Expected: ", training_data_output[i], ", Got: ", prediction)

# end of neural network layer classes
####################################################################################

### set up neural networks for AR application ###
model_fist_open = Network([
    Input("relu"),
    Convolution(16, (9,9), "relu"),
    Maxpooling((3,3),2),
    Convolution(32, (3,3), "relu"),
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(100, "relu"),
    Output(2, "softmax")])

model_ltr_directions = Network([
    Input("relu"),
    Convolution(16, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Convolution(32, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(25, "relu"),
    Output(3, "softmax")])

model_down_side_up = Network([
    Input("relu"),
    Convolution(16, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Convolution(32, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(50, "relu"),
    Output(3, "softmax")])

model_down_forwards_up = Network([
    Input("relu"),
    Convolution(8, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Convolution(16, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(50, "relu"),
    Output(3, "softmax")])

# load weights from pre-trained TF NN
def load_parameters():
    global model_fist_open, model_directions, model_down_side_up
    
    model_fist_open.layers[1].filters = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_open_fist/conv1_weights.npy")
    model_fist_open.layers[1].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_open_fist/conv1_biases.npy")
    model_fist_open.layers[3].filters = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_open_fist/conv2_weights.npy")
    model_fist_open.layers[3].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_open_fist/conv2_biases.npy")
    model_fist_open.layers[6].weights = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_open_fist/hidden_weights.npy")
    model_fist_open.layers[7].weights = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_open_fist/output_weights.npy")
    model_fist_open.layers[6].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_open_fist/hidden_biases.npy")
    model_fist_open.layers[7].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_open_fist/output_biases.npy")
    model_fist_open.networkInitialized = True

    model_ltr_directions.layers[1].filters = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_left_towards_right/conv1_weights.npy")
    model_ltr_directions.layers[1].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_left_towards_right/conv1_biases.npy")
    model_ltr_directions.layers[3].filters = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_left_towards_right/conv2_weights.npy")
    model_ltr_directions.layers[3].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_left_towards_right/conv2_biases.npy")
    model_ltr_directions.layers[6].weights = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_left_towards_right/hidden_weights.npy")
    model_ltr_directions.layers[7].weights = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_left_towards_right/output_weights.npy")
    model_ltr_directions.layers[6].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_left_towards_right/hidden_biases.npy")
    model_ltr_directions.layers[7].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_left_towards_right/output_biases.npy")
    model_ltr_directions.networkInitialized = True

    model_down_side_up.layers[1].filters = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_side_up/conv1_weights.npy")
    model_down_side_up.layers[1].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_side_up/conv1_biases.npy")
    model_down_side_up.layers[3].filters = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_side_up/conv2_weights.npy")
    model_down_side_up.layers[3].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_side_up/conv2_biases.npy")
    model_down_side_up.layers[6].weights = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_side_up/hidden_weights.npy")
    model_down_side_up.layers[7].weights = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_side_up/output_weights.npy")
    model_down_side_up.layers[6].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_side_up/hidden_biases.npy")
    model_down_side_up.layers[7].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_side_up/output_biases.npy")
    model_down_side_up.networkInitialized = True

    model_down_forwards_up.layers[1].filters = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_forwards_up/conv1_weights.npy")
    model_down_forwards_up.layers[1].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_forwards_up/conv1_biases.npy")
    model_down_forwards_up.layers[3].filters = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_forwards_up/conv2_weights.npy")
    model_down_forwards_up.layers[3].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_forwards_up/conv2_biases.npy")
    model_down_forwards_up.layers[6].weights = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_forwards_up/hidden_weights.npy")
    model_down_forwards_up.layers[7].weights = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_forwards_up/output_weights.npy")
    model_down_forwards_up.layers[6].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_forwards_up/hidden_biases.npy")
    model_down_forwards_up.layers[7].bias = np.load("../Neural Network/tensorflow_gesture_detection_kinect_side_down_forwards_up/output_biases.npy")
    model_down_forwards_up.networkInitialized = True

load_parameters()
# fist_time = []
# perform classification using the 3 gesture NNs
def handClassifiers(handImage):
    # global fist_time
    # s_time = time.time()
    fist_prediction = model_fist_open.predict(handImage)
    # e_time = time.time()
    # fist_time.append(e_time-s_time)
    # if(len(fist_time) == 100):
        # average_fist_time = np.mean(fist_time)
        # print("Fist classifier time: ", average_fist_time, ": ", 1/(average_fist_time), " fps")
        # exit()
    fist_index_min = np.argmax(fist_prediction)

    ltr_prediction = model_ltr_directions.predict(handImage)
    ltr_index_min = np.argmax(ltr_prediction)
    
    down_side_up_prediction = model_down_side_up.predict(handImage)
    down_side_up_index_min = np.argmax(down_side_up_prediction)

    down_forwards_up_prediction = model_down_forwards_up.predict(handImage)
    down_forwards_up_index_min = np.argmax(down_forwards_up_prediction)
    # e_time = time.time()
    # print("Classifier time: ", e_time-s_time, " =", 1/(e_time-s_time), " fps" )

    return fist_index_min , ltr_index_min, down_side_up_index_min, down_forwards_up_index_min

# Find the surface of the table with an initial normal vector and RANSAC-plane-finding algorithm
def surface_calibration():
    counter_new = 0
    normal_list = []
    # read in from Kinect
    videoInput = freenect.sync_get_video()[0]
    depthInput = np.double(freenect.sync_get_depth()[0])
    print("Read in empty environment data from Kinect")
    print("Performing calibration...")
    # find normal of each pixel to the plane it rests on
    @nb.jit(nopython=True) # accelerate tedious for loops using Numba
    def return_normals(depthIn):
        normals = np.zeros((480,640,3))
        for x in range(1,depthInput.shape[0]-1,1):
            for y in range(1,depthInput.shape[1]-1,1):
                if(depthInput[x][y] == 2047):
                    normals[x][y] = 0
                else:
                    # calculate derivatives of pixels based on xy planes of pixels around it
                    dzdx = np.double(depthInput[x+1][y] - depthInput[x-1][y]) / 2
                    dzdy = (depthInput[x][y+1] - depthInput[x][y-1]) / 2
                    vector_d = [-dzdx, -dzdy, 1.0]
                    vector_mag = np.sqrt(vector_d[0]**2 + vector_d[1]**2 + vector_d[2]**2)
                    new_normal = np.array(vector_d)/vector_mag # normalize into unit vectors
                    normals[x][y] = new_normal
        return normals
    normals = return_normals(depthInput) # normals to plane each pixel rests on

    # # find all the pixels that have a normal that matches the table surface normal
    # def find_plane(normals):
    #     chosen_normal = normals[410][360] # value of ceiling-facing object found in calibrated rectangle
    #     normals_detected = np.zeros((normals.shape)) # coloured output plane representing where similar normals found
    #     for x in range(1,depthInput.shape[0]-1,1):
    #         for y in range(1,depthInput.shape[1]-1,1):
    #             if(np.allclose(normals[x][y], chosen_normal, 0.2,0.2) == True):
    #                 normals_detected[x][y] = 255
    #     return normals_detected
    
    # normals_detected = find_plane(normals)
    # cv2.imshow("normals detected", normals_detected)
    
    # cv2.waitKey(10000)

    # get all pixels of surface by same colour as original normal (RANSAC-inspired algorithm)
    # @nb.jit(nopython=True) # accelerate tedious for loops using Numba
    def RANSAC_get_surface_pixels():
        starting_surface_point_row_coord = 360 # row of normal where correct table surface found (specificed by calibration)
        starting_surface_point_col_coord = 410 # col of normal where correct table surface found (specificed by calibration)
        surface_pixel_rows = [360] # all the rows where the table surface exists
        surface_pixel_cols = [410] # all the cols where the table surface exists
        first_coord = str(starting_surface_point_col_coord) + "," + str(starting_surface_point_row_coord)
        surface_pixel_coords=[first_coord] # string coords of where the table surface exists - for prevention of checking the same coordinates multiple times
        
        # chosen_normal = [0.9,-0.2,0.6]
        chosen_normal = normals[410][360]
        pixel_counter = 0 # how many pixels are part of the table surface
        # RANSAC algorithm - grow surface region pixel-by-pixel if colour close to original surface pixel colour
        while(pixel_counter<len(surface_pixel_rows)):
            current_row_coord = surface_pixel_rows[pixel_counter]
            current_col_coord = surface_pixel_cols[pixel_counter]
            new_coord_1 = str(current_col_coord+1) + "," + str(current_row_coord)
            if(current_col_coord+1 < 480): # check pixel to right and not off screen
                # check if mean of RGB values close to original surface pixel colour
                # also check if coordinate hasn't already been added to surface
                if(np.allclose(normals[current_col_coord+1][current_row_coord], chosen_normal,0.3,0.3) and new_coord_1 not in surface_pixel_coords):
                    surface_pixel_rows.append(current_row_coord) # add pixel row, col and string identifier to arrays to add it to the table surface
                    surface_pixel_cols.append(current_col_coord+1)
                    surface_pixel_coords.append(new_coord_1)
            new_coord_2 = str(current_col_coord-1) + "," + str(current_row_coord)
            if(current_col_coord-1 > -1 ): # check pixel to left and not off screen
                if(np.allclose(normals[current_col_coord-1][current_row_coord], chosen_normal,0.3,0.3) and new_coord_2 not in surface_pixel_coords):
                    surface_pixel_rows.append(current_row_coord)
                    surface_pixel_cols.append(current_col_coord-1)
                    surface_pixel_coords.append(new_coord_2)
            new_coord_3 = str(current_col_coord) + "," + str(current_row_coord+1)
            if(current_row_coord+1 < 640): # check pixel below and not off screen
                if(np.allclose(normals[current_col_coord][current_row_coord+1], chosen_normal,0.3,0.3) and new_coord_3 not in surface_pixel_coords):
                    surface_pixel_rows.append(current_row_coord+1)
                    surface_pixel_cols.append(current_col_coord)
                    surface_pixel_coords.append(new_coord_3)
            new_coord_4 = str(current_col_coord) + "," + str(current_row_coord-1)
            if(current_row_coord-1 > -1): # check pixel above and not off screen
                if(np.allclose(normals[current_col_coord][current_row_coord-1], chosen_normal,0.3,0.3) and new_coord_4 not in surface_pixel_coords):
                    surface_pixel_rows.append(current_row_coord-1)
                    surface_pixel_cols.append(current_col_coord)
                    surface_pixel_coords.append(new_coord_4)
            pixel_counter += 1
        print("Found surface pixels: ", pixel_counter)

        # Set 0's and 1's in a matrix for the detected surface - for easy checking if a pixel is part of the surface and visualization
        detected_surface = np.zeros(videoInput.shape)
        for each_pixel in range(len(surface_pixel_rows)):
            detected_surface[surface_pixel_cols[each_pixel]][surface_pixel_rows[each_pixel]] = 1
        
        return detected_surface
    
    detected_surface = RANSAC_get_surface_pixels()

    # return the original environment RGB image, depth image, detected surface matrix and normals image
    return videoInput, depthInput, detected_surface, normals

# Check if the bottom of the cube is about to collide with the surface of the table
# @nb.jit(nopython=True) # accelerate tedious for loops using Numba
def surface_collision(startup_depthInput,startup_detected_surface, proposed_cube_x_coord,proposed_cube_y_coord, proposed_cube_depth):
    proposed_col = int((proposed_cube_y_coord/8)*480) # 0-8 top to bottom of screen
    proposed_row = int((proposed_cube_x_coord/11)*640) # 0-11 left to right of screen 
    # print("proposed_cube_x_coord",proposed_cube_x_coord)
    # print("proposed_cube_y_coord: ", proposed_cube_y_coord)
    # print("proposed_row: ", proposed_row)
    # print("proposed_col: ", proposed_col)
    # prevent indexing out of frame
    if(proposed_row<15):
        proposed_row +=15
    if(proposed_row > 474):
        proposed_row -= 15
    if(proposed_col<15):
        proposed_col +=15
    if(proposed_col > 634):
        proposed_col -= 15
    # wrap in if statement to see if proposed coords even in surface matrix
    if(1 in startup_detected_surface[proposed_col-15:proposed_col+15,proposed_row-15:proposed_row+15]): # surface pixel close to where the cube will be
        # find closest depth value to proposed depth value in a 5x5 region
        close_region = startup_depthInput[proposed_col-15:proposed_col+15,proposed_row-5:proposed_row+15]
        close_region_flattened = close_region.flatten()
        close_depth_delta_flattened = np.abs(close_region-proposed_cube_depth).flatten() # find closest depth value to desired value
        closest_depth_value_in_region = close_region_flattened[close_depth_delta_flattened.argmin()]
        # print("proposed_cube_depth: ", proposed_cube_depth)
        # print("actual depths: ", close_region_flattened)
        if(abs(closest_depth_value_in_region-proposed_cube_depth)  < 100 ): # check if the cube depth is anywhere near the surface depth
            return "collision"
    return "nocollision"

# Rotate cube according to classifiers
def rotate_cube(ltr_directions_index_min,down_side_up_index_min,down_forwards_up_index_min):
    global x_rotation_degrees, y_rotation_degrees, z_rotation_degrees

    # Down side up classifier (z axis rotation)
    if(down_side_up_index_min == 0): # down
        glRotatef(-5,0,0,1) # rotate z by -5 degrees
        z_rotation_degrees -= 5
        rotation_history.append(["z",-5])
        # print("Down")
    if(down_side_up_index_min == 1): # side
        # Leave rotation as is
        # print("Side")
        pass
    if(down_side_up_index_min == 2): # up
        glRotatef(5,0,0,1) # rotate z by 5 degrees
        z_rotation_degrees += 5
        rotation_history.append(["z",5])
        # print("Up")

    # Left towards right classifier (y axis rotation)
    if(ltr_directions_index_min == 0): # left
        glRotatef(-5,0,1,0) # rotate y by -5 degrees
        y_rotation_degrees -= 5
        rotation_history.append(["y",-5])
        # print("Left")
    if(ltr_directions_index_min == 1): # towards
        # leave rotation as is
        # print("Towards")
        pass
    if(ltr_directions_index_min == 2): # right
        glRotatef(5,0,1,0) # rotate y by 5 degrees
        y_rotation_degrees += 5
        rotation_history.append(["y",5])
        # print("Right")
            
    # Down forwards up classifier (x axis rotation)
    if(down_forwards_up_index_min == 0): # down
        glRotatef(-5,1,0,0) # rotate x by -5 degrees
        x_rotation_degrees -= 5
        rotation_history.append(["x",-5])
        # print("Down")
    if(down_forwards_up_index_min == 1): # forwards
        # leave rotation as is
        # print("Forwards")
        pass
    if(down_forwards_up_index_min == 2): # up
        glRotatef(5,1,0,0) # rotate x by 5 degrees
        x_rotation_degrees += 5
        rotation_history.append(["x",5])
        # print("Up")

# start a pygame window and modify it for displaying OpenGL content - the main AR UI
def main():
    global previous_depth, cube_scale, cube_x_coord, cube_y_coord, y_rotation_degrees, x_rotation_degrees
    pygame.init() 
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) # clear OpenGL canvas

    gluPerspective(45, (display[0]/display[1]), 0.01, 50.0) # set perspective of OpenGL window
    # 45deg fov, aspect ratio, near + far clipping planes (where cube visible)
    glTranslatef(-5.0,3.5, -10.0) # original translation to get cube in starting position

    # surface calibration to find surface of table at start of program
    print("Calibration: Surface detection in progress...")
    use_preloaded_calibration = True
    if(use_preloaded_calibration): 
        # load in the calibration from when first setting up the camera in its current position
        startup_videoInput = np.load("calibrated_startup_videoInput.npy")
        startup_depthInput = np.load("calibrated_startup_depthInput.npy")
        startup_detected_surface = np.load("calibrated_startup_detected_surface.npy")
        startup_normals = np.load("calibrated_startup_normals.npy")
    else:
        # or calibrate using the Kinect now with new data. ETA runtime ~100s
        desk_in_place = False;
        while(desk_in_place == False):
            videoInput = cv2.cvtColor(freenect.sync_get_video()[0],cv2.COLOR_BGR2RGB)
            cv2.putText(videoInput,'Ensure rectangle is placed over table surface and press SPACE to calibrate.', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),2,2)
            cv2.imshow("Calibration", cv2.rectangle(videoInput,(350,400),(370,420), (0,0,255), 2))
            cv2.waitKey(1)
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    # start calibration when SPACE pressed
                    if event.key == pygame.K_SPACE:
                        desk_in_place = True
                        cv2.destroyAllWindows()
                        for i in range (1,5):
                            cv2.waitKey(1)
        startup_videoInput, startup_depthInput, startup_detected_surface, startup_normals = surface_calibration()
        np.save("calibrated_startup_videoInput", startup_videoInput)
        np.save("calibrated_startup_depthInput", startup_depthInput)
        np.save("calibrated_startup_detected_surface", startup_detected_surface)
        np.save("calibrated_startup_normals", startup_normals)
    
    # run surface detection once at start to initialise Numba preprocessing
    collision_status_temp = surface_collision(startup_depthInput, startup_detected_surface,0,0, 0) 

    cv2.imshow("Normals", startup_normals)
    cv2.imshow("Detected surface", startup_detected_surface)
    print("Calibration complete.")
    
    # AR application
    while True:
        start_ar_time = time.time()
        # setup background image textures
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        texture_background = glGenTextures(1) # create OpenGL texture
        
        # read in RGB + depth data from Kinect
        videoInput = freenect.sync_get_video()[0]
        depthInput = freenect.sync_get_depth()[0] # depthInput.shape = (480, 640)
        # cv2.imshow("Depth Input", cv2.applyColorMap((depthInput).astype(np.uint8), cv2.COLORMAP_JET))
        original_image = np.array(videoInput)
        background_image = cv2.flip(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB ), 0) # convert BGR to RGB colour space
        # get info of frame and convert to format needed for OpenGL texture
        background_image = Image.fromarray(background_image)
        image_width = background_image.size[0]
        image_height = background_image.size[1]
        background_image = background_image.tobytes('raw', 'BGRX', 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, background_image)

        # draw background image on a quad
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 1.0,  1.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,  1.0, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        
        glDisable(GL_DEPTH_TEST)
    
        # check for end/exit program command
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                pass
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # perform hand tracking on image
        hand_present , hand_depth , hand_bounding_box, extracted_hand, side_entered_from = handTracking(original_image,depthInput)
        # print("hand_depth: ", hand_depth)
        # print("hand_bounding_box: ", hand_bounding_box)
        # print("handboundbox: ", hand_bounding_box)
        
        collisions = [0,0,0,0] # no collisions at start

        # get depth values around hand region for object detection/collision avoidance
        if(hand_depth != 2047 and hand_depth != -1):
            depth_boxes, depthInput = depth_around_object(depthInput, hand_bounding_box, side_entered_from)
            
            if(abs(depth_boxes[0]-hand_depth)<30): 
                if(side_entered_from != 0):
                    collisions[0] = 1
                # print("Collision at top!")
            if(abs(depth_boxes[1]-hand_depth)<30):
                if(side_entered_from != 1):
                    collisions[1] = 1
                # print("Collision at left!")
            if(abs(depth_boxes[2]-hand_depth)<30):
                if(side_entered_from != 2):
                    collisions[2] = 1
                # print("Collision at bottom!")
            if(abs(depth_boxes[3]-hand_depth)<30):
                if(side_entered_from != 3):
                    collisions[3] = 1
                # print("Collision at right!")
            
            # Display depth map
            cv2.imshow("Depth Input", cv2.applyColorMap((depthInput).astype(np.uint8), cv2.COLORMAP_JET))
        
        # if a hand has been detected, move the cube and rotate it as necessary
        if(hand_present):

            # start_time = time.time()
            # perform gesture recognition on image of hand
            fist_index_min , ltr_directions_index_min, down_side_up_index_min, down_forwards_up_index_min = handClassifiers(extracted_hand) 
            # print(f"Time for inferrance: {(time.time()  - start_time):.4f} seconds")
            if(fist_index_min==0): # open hand
                print("Open")
                rotate_cube(ltr_directions_index_min,down_side_up_index_min,down_forwards_up_index_min) # rotate cube if hand open
                pass
            if(fist_index_min==1): # fist
                # move and scale cube if hand close to it
                print("Fist")
                move_and_scale_cube(startup_depthInput, startup_detected_surface, hand_bounding_box[0]/640, hand_bounding_box[1]/480, hand_depth,collisions)

        # manual rotation of cube for testing purposes
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # upon close of window
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # control cube with keyboard input
                if event.key == pygame.K_LEFT:
                    y_rotation_degrees += 5
                    # glTranslatef(-1.0,-1.0,-1.0)
                    glRotatef(5,0,1,0)    
                    # glTranslatef(1.0,1.0,1.0)
                if event.key == pygame.K_RIGHT:
                    y_rotation_degrees -= 5
                    # glTranslatef(-1.0,-1.0,-1.0)
                    glRotatef(-5,0,1,0)    
                    # glTranslatef(1.0,1.0,1.0)
                if event.key == pygame.K_w:
                    # glTranslatef(-1.0,-1.0,-1.0)
                    glRotatef(5,1,0,0)
                    x_rotation_degrees +=5
                    # glTranslatef(1.0,1.0,1.0)
                if event.key == pygame.K_s:
                    # glTranslatef(-1.0,-1.0,-1.0)
                    glRotatef(-5,1,0,0)
                    x_rotation_degrees -= 5
                    # glTranslatef(1.0,1.0,1.0)
                if event.key == pygame.K_a:
                    glTranslatef(-1.0,-1.0,-1.0)
                    glRotatef(45,0,0,1)
                    glTranslatef(1.0,1.0,1.0)
                if event.key == pygame.K_d:
                    glTranslatef(-1.0,-1.0,-1.0)
                    glRotatef(45,0,0,-1)
                    glTranslatef(1.0,1.0,1.0)

        Cube() # display the cube
        # print("X: ", cube_x_coord, ", Y: ", cube_y_coord, ", Z: ", cube_z_coord, ", Scale: ", cube_scale)
        pygame.display.flip() # update the display
        # if cv2.waitKey(1) == ord(" "):
        #     break    
        end_time = time.time()
        print("Runtime for loop: ", (end_time-start_ar_time), " : ", 1/(end_time-start_ar_time), " fps")

# actually run the AR application
main()
