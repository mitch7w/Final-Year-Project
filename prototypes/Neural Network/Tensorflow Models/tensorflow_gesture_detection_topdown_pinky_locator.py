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
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 
# set random seed for random number generator
rng = default_rng(777)

def loadImage(imageName):
    if(isinstance(imageName,str)):
        # load image if just it's name was passed in
        imageName = cv2.imread(imageName,1) 
    # imageName is now the RGB-passed-in image
    # perform skin segmentation using the Ycbcr color space
    # put in tracker code here
    image = cv2.resize(imageName, (320,240), interpolation = cv2.INTER_AREA)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
    # print(image_ycrcb[0][0])
    skin1 = (50, 89, 136)
    skin2 = (231, 147, 181)
    mask = cv2.inRange(image_ycrcb, skin1, skin2)
    result = cv2.bitwise_and(image, image, mask=mask)
    # # do contouring and return that instead
    # contours = []
    # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # if(len(contours) != 0):
    #     contours = max(contours, key=lambda x: cv2.contourArea(x))
    # empty_image = np.zeros((320, 240, 3))    
    # cv2.drawContours(empty_image, [contours], -1, (255, 255, 0), 2)
    # return np.array(empty_image)
    # contours = contours.flatten()
    # while(len(contours) < 1000*2):
    #     contours = np.append(contours, 0)
    # contours = contours.reshape((1000,1,2))
    # return np.array(contours)
    # # find highest concentration of values in mask matrix closest to bottom of image
    # r_region_counters = []
    # c_region_counters = []
    # region_sums = []
    # south_deltas = []
    # # find all regions with pixel count larger than 270 000
    # for r_counter in range(0,320,5):
    #     for c_counter in range(0,240,5):
    #         region_sum = np.sum(mask[r_counter:r_counter+60,c_counter:c_counter+80]) # sum all white pixels in proposal region
    #         if(region_sum > 100000): # if more white pixels here than average
    #             south_delta = abs(r_counter-240)
    #             south_deltas.append(south_delta)
    #             region_sums.append(region_sum)
    #             r_region_counters.append(r_counter)
    #             c_region_counters.append(c_counter)
    # # find southmost
    # if(len(south_deltas) ==0):
    #     print("error no southmost delta!")
    # southmost_delta = np.amin(south_deltas)
    # # find largest region within 50px of southmost point
    # largest_south_index = 0
    # for current_region in range(len(region_sums)):
    #     # print("south_deltas[current_region] - southmost_delta: ", south_deltas[current_region] - southmost_delta)
    #     if(region_sums[current_region] > region_sums[largest_south_index]):
    #         if( abs(south_deltas[current_region] - southmost_delta) < 50):
    #             largest_south_index = current_region
    # extracted_hand = image[r_region_counters[largest_south_index]:r_region_counters[largest_south_index]+60 , c_region_counters[largest_south_index]:c_region_counters[largest_south_index]+80]
    loadedImage = np.array(result)/255 # divide by 255 in order to normalize input
    if(loadedImage.shape[2] !=3): # if it's not RGB put it in a 3D shape
        new3DImage = np.empty((loadedImage.shape[0],loadedImage.shape[1],1))
        for x in range(len(new3DImage)):
            for y in range(len(new3DImage[x])):
                for i in range(len(new3DImage[x][y])):
                    new3DImage[x][y][i] = loadedImage[x][y]
        loadedImage = new3DImage
    return np.array(loadedImage)

# load images
training_data_input = []
for imageNumber in range(1,56):
    image_pathname = "/Users/mitch/Documents/University/Project/pinky_locator/gesture" + str(imageNumber) + ".jpg"
    training_data_input.append(loadImage(image_pathname))
training_data_input = np.array(training_data_input)
print("training_data_input shape: ", np.array(training_data_input).shape)

# load output classes
training_data_output = []
for imageNumber in range(1,56):
    regions_filepath = "/Users/mitch/Documents/University/Project/pinky_locator/region" + str(imageNumber) + ".npy"
    regions = np.load(regions_filepath)
    training_data_output.append(regions.flatten())
print("training_data_output shape: ", np.array(training_data_output).shape)

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data
x_train = np.array(training_data_input[:-5])
y_train = np.array(training_data_output[:-5])
x_test = np.array(training_data_input[-5:])
y_test = np.array(training_data_output[-5:])
print("x_train.shape: ", x_train.shape)

num_output_classes = 3072
input_shape = training_data_input[0].shape
print(input_shape)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(16, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
        # layers.Conv2D(32, kernel_size=(2,2), activation="relu"),
        # layers.MaxPooling2D(pool_size=(3,3)),
        layers.Flatten(),
        layers.Dense(1000, activation="relu"),
        layers.Dense(num_output_classes, activation="softmax"), 
    ]
)

model.summary()

#Train the model
batch_size = 1
epochs = 20

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

# predict using webcam
cap = cv2.VideoCapture(1)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    cv2.imshow('Input camera', cv2.flip(image, 1))
    # do some preprocessing before prediction
    loaded_image_new = np.array(loadImage(image)).reshape(1,input_shape[0],input_shape[1],input_shape[2])
    cv2.imshow('Extracted Hand', loaded_image_new[0])
    new_prediction = model.predict(loaded_image_new)
    index_min = np.argmax(new_prediction[0])
    if(index_min == 0):
        print("Prediction: Open")
    else:
        print("Prediction: Fist")
    if cv2.waitKey(5) == ord(" "):
        break

cap.release()


