# -*- coding: utf-8 -*-

# Setup
import numpy as np
from tensorflow import keras
import tensorflow as tf
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

# hand detection and tracking algorithms

def loadImageSmall(background_image):
    if(isinstance(background_image,str)):
        # load image if just it's name was passed in
        background_image = cv2.imread(background_image,1) 
    # read in from Kinect
    background_image = np.array(background_image)
    image = cv2.resize(background_image, (80,60), interpolation = cv2.INTER_AREA)
    
    cv2.imshow("Image", image)
    # increase saturation
    im_pil = Image.fromarray(image)
    converter = ImageEnhance.Color(im_pil)
    img2 = converter.enhance(1.5)
    # For reversing the operation:
    image = np.asarray(img2)
    # cv2.imshow("Enhancer", image)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
    # cv2.imshow("Enhancer", image)
    # print(image_ycrcb[0][0])
    # skin1 = (0, 100, 136)
    # skin2 = (255, 144, 156)
    skin1 = (114,74,147)
    skin2 = (255,190,204)
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
    for c_counter in range(0,80):
        for r_counter in range(0,60):
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
    max_sum = -1
    max_c = -1
    max_r = -1

    # if there is actually a region that meets the required concentration of pixels
    if(len(edge_deltas) !=0):
        closest_delta_index = np.argmin(edge_deltas)
        crop_coords = [c_region_counters[closest_delta_index],c_region_counters[closest_delta_index],r_region_counters[closest_delta_index],r_region_counters[closest_delta_index]]
        # depthFeed = frame_convert2.pretty_depth_cv(depthInput) # modified depthInput too
        # depthFeedColoured = cv2.applyColorMap(depthFeed,cv2.COLORMAP_JET)
        # depthFeedColoured.shape = (480, 640, 3)
        
        # rectangle (x1,y1) (x2,y2) left to right, top to bottom
        # depthInput[0-480][0-640] # top to bottom, left to right
        # (480, 640)

        # now find largest region within 10 regions of this region.
        largest_region_index = -1
        c_range = [c_region_counters[closest_delta_index] -10 , c_region_counters[closest_delta_index] +10 ]
        r_range = [r_region_counters[closest_delta_index] -10 , r_region_counters[closest_delta_index] +10 ]
        if(c_range[0] < 0): c_range[0] = 0
        if(c_range[1] > 79): c_range[1] = 79
        if(r_range[0] < 0): r_range[0] = 0
        if(r_range[1] > 59): r_range[1] = 59
        
        for c_counter in range(c_range[0],c_range[1]):
            for r_counter in range(r_range[0],r_range[1]):
                region_sum = np.sum(mask[r_counter:r_counter+15,c_counter:c_counter+20]) # sum all white pixels in proposal region
                if(region_sum > max_sum):
                    max_sum = region_sum
                    max_c = c_counter
                    max_r = r_counter
        # print("c_counter: ", c_region_counters[closest_delta_index], ", r_region: ", r_region_counters[closest_delta_index])
        crop_coords = [max_c,max_c+20,max_r,max_r+15] # modify crop coords to desired region

    # widen extracted window to get the whole hand in the frame
        if(crop_coords[0]-10 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 10

        if(crop_coords[1]+10 >79): # how much extra right to do
            crop_coords[1] = 79
        else:
            crop_coords[1] += 10

        if(crop_coords[2] - 10 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 10

        if(crop_coords[3]+10 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 10

    # extracted_hand = image[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]] 
    extracted_hand = result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]] # return Ycbcr-masked image
    # print("extracted_hand.shape: ", extracted_hand.shape)
    if(extracted_hand.shape != (35, 40, 3)):
        extracted_hand = cv2.resize(result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]], (40,35), interpolation = cv2.INTER_AREA)
    
    # TODO large image processing
    # increase saturation
    im_pil_1 = Image.fromarray(background_image)
    converter_1 = ImageEnhance.Color(im_pil_1)
    img2_1 = converter_1.enhance(1.5)
    image_1 = np.asarray(img2_1)
    # cv2.imshow("Enhancer", image)
    image_ycrcb_1 = np.array(cv2.cvtColor(image_1, cv2.COLOR_RGB2YCR_CB)) # convert to Ycbcr colour space
    mask1 = cv2.inRange(image_ycrcb_1, skin1, skin2)
    result1 = cv2.bitwise_and(background_image, background_image, mask=mask1)
    extracted_hand = cv2.resize(result1, (320,240), interpolation=cv2.INTER_AREA) # TODO remove
    
    
    # cv2.imshow("extracted_hand_rectangle", cv2.rectangle(image, (max_c,max_r), (max_c+20,max_r+15), (0,0,255), 2))
    # cv2.waitKey(1)
    # make sure all images are same size even if cropped on the side
    loadedImage = np.array(extracted_hand)/255 # divide by 255 in order to normalize input
    if(len(loadedImage.shape) == 2): # if it's not 3 channels long put it in a 3D shape
        loadedImage = loadedImage.reshape((loadedImage.shape[0],loadedImage.shape[1],1))
    return np.array(loadedImage)

# load images
training_data_input = []
# for i in range(1,1401):
for i in range(1,1401):
    imageName = "/Users/mitch/Documents/University/Project/open_fist_close/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImageSmall(imageName))
print("training_data_input shape: ", np.array(training_data_input).shape)

training_data_output = []

# load output classes
training_data_output = []
for j in range(1):
    for i in range(700):
            training_data_output.append([1,0]) # Open
    for i in range(700):
        training_data_output.append([0,1]) # Fist
    
print("training_data_output shape: ", np.array(training_data_output).shape)

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data

x_train = np.array(training_data_input[:-140])
y_train = np.array(training_data_output[:-140])
x_test = np.array(training_data_input[-140:])
y_test = np.array(training_data_output[-140:])
print("x_train.shape: ", x_train.shape)

num_output_classes = 2
input_shape = training_data_input[0].shape

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(9,9), activation="relu"),
        layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
        layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(num_output_classes, activation="softmax"), 
    ]
)

model.summary()

#Train the model
batch_size = 1
epochs = 6

# Setup model
optimizer = keras.optimizers.Adam(lr=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# # Early stopping
# callback = callbacks.EarlyStopping(monitor='accuracy', patience=1)
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[callback],verbose=1)

# train model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=1)

# plot accuracy and loss curves

# from matplotlib import pyplot as plt

# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='lower right')
# plt.title("")
# plt.show()

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.title("")
# plt.legend(['train', 'val'], loc='upper right')
# plt.show()

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# # print("Convolutional Layer")
np.save("../model_weights/tensorflow_gesture_detection_kinect_side_open_fist_close/conv1_weights", model.layers[0].get_weights()[0])
np.save("../model_weights/tensorflow_gesture_detection_kinect_side_open_fist_close/conv1_biases", model.layers[0].get_weights()[1])
np.save("../model_weights/tensorflow_gesture_detection_kinect_side_open_fist_close/conv2_weights", model.layers[2].get_weights()[0])
np.save("../model_weights/tensorflow_gesture_detection_kinect_side_open_fist_close/conv2_biases", model.layers[2].get_weights()[1])
# np.save("tensorflow_gesture_detection_kinect_side_open_fist/conv3_weights", model.layers[4].get_weights()[0])
# np.save("tensorflow_gesture_detection_kinect_side_open_fist/conv3_biases", model.layers[4].get_weights()[1])

# print("Hidden Layer")
np.save("../model_weights/tensorflow_gesture_detection_kinect_side_open_fist_close/hidden_weights", model.layers[5].get_weights()[0])
np.save("../model_weights/tensorflow_gesture_detection_kinect_side_open_fist_close/hidden_biases", model.layers[5].get_weights()[1])
# print(np.array(model.layers[3].get_weights()[0]).shape)

# print("Output Layer")
np.save("../model_weights/tensorflow_gesture_detection_kinect_side_open_fist_close/output_weights", model.layers[6].get_weights()[0])
np.save("../model_weights/tensorflow_gesture_detection_kinect_side_open_fist_close/output_biases", model.layers[6].get_weights()[1])
print("Saved weights.")

# predict using Kinect input
while True:
    videoInput = freenect.sync_get_video()[0]
    videoFeed = frame_convert2.video_cv(videoInput)
    depthInput = freenect.sync_get_depth()[0]
    loaded_image_new = np.array(loadImageLarge(videoFeed)).reshape(1,input_shape[0],input_shape[1],input_shape[2])
    cv2.imshow('Extracted Hand', loaded_image_new[0])
    new_prediction = model.predict(loaded_image_new, verbose=0)
    
    max_pred = np.amax(new_prediction[0])
    if(max_pred < 0.94): # threshold low values
        continue
    print("new_prediction: ", new_prediction, " ", end=" ")
    index_max = np.argmax(new_prediction[0])
    if(index_max == 0):
        print("Prediction: Open")
    if(index_max == 1):
        print("Prediction: Fist")
    if cv2.waitKey(5) == ord(" "):
        break
