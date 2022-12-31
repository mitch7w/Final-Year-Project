# -*- coding: utf-8 -*-

# Setup
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import scipy.signal
import cv2
from numpy.random import default_rng
import random
import sys
import freenect
from PIL import ImageEnhance, Image
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 
# set random seed for random number generator
rng = default_rng(777)

def loadImage(background_image):
    if(isinstance(background_image,str)):
        # load image if just it's name was passed in
        background_image = cv2.imread(background_image,1) 
    # read in from Kinect
    image = cv2.resize(background_image, (320,240), interpolation = cv2.INTER_AREA)
    cv2.imshow("Image", cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
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

    extracted_hand = result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]] 
    if(extracted_hand.shape != (140, 180, 3)):
        extracted_hand = cv2.resize(result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]], (180,140), interpolation = cv2.INTER_AREA)
    extracted_hand_greyscale = cv2.cvtColor(extracted_hand,cv2.COLOR_RGB2GRAY)
    # make sure all images are same size even if cropped on the side
    loadedImage = np.array(extracted_hand_greyscale)/255 # divide by 255 in order to normalize input
    if(len(loadedImage.shape) == 2): # if it's not 3 channels long put it in a 3D shape
        loadedImage = loadedImage.reshape((loadedImage.shape[0],loadedImage.shape[1],1))
    return np.array(loadedImage)

# load images
training_data_input = []
for i in range(1,271):
    imageName = "/Users/mitch/Documents/University/Project/kinect_side_palm_side_back/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImage(imageName))
print("training_data_input shape: ", np.array(training_data_input).shape)

# load output classes
training_data_output = []
for i in range(90):
        training_data_output.append([1,0,0]) # Back
for i in range(90):
    training_data_output.append([0,1,0]) # Side
for i in range(90):
    training_data_output.append([0,0,1]) # Palm
print("training_data_output shape: ", np.array(training_data_output).shape)

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data

x_train = np.array(training_data_input[:-27])
y_train = np.array(training_data_output[:-27])
x_test = np.array(training_data_input[-27:])
y_test = np.array(training_data_output[-27:])
print("x_train.shape: ", x_train.shape)

num_output_classes = 3
input_shape = training_data_input[0].shape
print(input_shape)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
        layers.Flatten(),
        layers.Dense(500, activation="relu"),
        layers.Dense(num_output_classes, activation="softmax"), 
    ]
)

model.summary()

#Train the model
batch_size = 1
epochs = 5

# Setup model
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# # Early stopping
# callback = callbacks.EarlyStopping(monitor='accuracy', patience=1)
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[callback],verbose=1)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=1)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

print("Convolutional Layer")
np.save("tensorflow_gesture_detection_kinect_side_palm_side_back/conv_weights", model.layers[0].get_weights()[0])
np.save("tensorflow_gesture_detection_kinect_side_palm_side_back/conv_biases", model.layers[0].get_weights()[1])

# print("Hidden Layer")
np.save("tensorflow_gesture_detection_kinect_side_palm_side_back/hidden_weights", model.layers[3].get_weights()[0])
np.save("tensorflow_gesture_detection_kinect_side_palm_side_back/hidden_biases", model.layers[3].get_weights()[1])
print(np.array(model.layers[3].get_weights()[0]).shape)

# print("Output Layer")
np.save("tensorflow_gesture_detection_kinect_side_palm_side_back/output_weights", model.layers[4].get_weights()[0])
np.save("tensorflow_gesture_detection_kinect_side_palm_side_back/output_biases", model.layers[4].get_weights()[1])

# predict using Kinect input
while True:
    videoInput = freenect.sync_get_video()[0]
    depthInput = freenect.sync_get_depth()[0]
    background_image = np.array(videoInput)
    loaded_image_new = np.array(loadImage(background_image)).reshape(1,input_shape[0],input_shape[1],input_shape[2])
    cv2.imshow('Extracted Hand', loaded_image_new[0])
    new_prediction = model.predict(loaded_image_new, verbose=0)
    index_min = np.argmax(new_prediction[0])
    if(index_min == 0):
        print("Prediction: Back")
    if(index_min == 1):
        print("Prediction: Side")
    if(index_min == 2):
        print("Prediction: Palm")
    if cv2.waitKey(5) == ord(" "):
        break

