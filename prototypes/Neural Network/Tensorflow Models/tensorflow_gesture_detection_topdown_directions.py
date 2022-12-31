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
    image = cv2.resize(imageName, (80,60), interpolation = cv2.INTER_AREA)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
    # print(image_ycrcb[0][0])
    skin1 = (50, 89, 136)
    skin2 = (231, 147, 181)
    mask = cv2.inRange(image_ycrcb, skin1, skin2)
    result = cv2.bitwise_and(image, image, mask=mask)
    # find highest concentration of values in mask matrix closest to bottom of image
    r_region_counters = []
    c_region_counters = []
    region_sums = []
    south_deltas = []
    # find all regions with pixel count larger than 270 000
    for r_counter in range(0,80,5):
        for c_counter in range(0,60,5):
            region_sum = np.sum(mask[r_counter:r_counter+15,c_counter:c_counter+20]) # sum all white pixels in proposal region
            if(region_sum > 1000): # if more white pixels here than average
                south_delta = abs(r_counter-60)
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
                if( abs(south_deltas[current_region] - southmost_delta) < 25):
                    largest_south_index = current_region
        crop_coords = [c_region_counters[largest_south_index],c_region_counters[largest_south_index],r_region_counters[largest_south_index],r_region_counters[largest_south_index]]
        if(crop_coords[0]-5 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 5

        if(crop_coords[1]+25 >79): # how much extra right to do
            crop_coords[1] = 319
        else:
            crop_coords[1] += 25

        if(crop_coords[2] - 10 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 10

        if(crop_coords[3]+20 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 20
    else:
        print("Error!: no regions with pixel concentration above threshold")
        crop_coords = crop_coords = [0,30,0,30]
    extracted_hand = result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]] 
    if(extracted_hand.shape != (30, 30, 3)):
        extracted_hand = cv2.resize(result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]], (30,30), interpolation = cv2.INTER_AREA)
    extracted_hand_greyscale = cv2.cvtColor(extracted_hand,cv2.COLOR_RGB2GRAY)
    # make sure all images are same size even if cropped on the side
    loadedImage = np.array(extracted_hand_greyscale)/255 # divide by 255 in order to normalize input
    if(len(loadedImage.shape) == 2): # if it's not 3 channels long put it in a 3D shape
        loadedImage = loadedImage.reshape((loadedImage.shape[0],loadedImage.shape[1],1))
    return np.array(loadedImage)

# load images
training_data_input = []
for i in range(1,211):
    imageName = "/Users/mitch/Documents/University/Project/topdown_directions/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImage(imageName))
print("training_data_input shape: ", np.array(training_data_input).shape)

# load output classes
training_data_output = []
for i in range(71):
    training_data_output.append([1,0,0]) # Up
for i in range(71):
    training_data_output.append([0,1,0]) # Left
for i in range(71):
    training_data_output.append([0,0,1]) # Right
print("training_data_output shape: ", np.array(training_data_output).shape)

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data

x_train = np.array(training_data_input[:-21])
y_train = np.array(training_data_output[:-21])
x_test = np.array(training_data_input[-21:])
y_test = np.array(training_data_output[-21:])
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
epochs = 3

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

# print("Convolutional Layer")
np.save("tensorflow_topdown_directions/conv_weights", model.layers[0].get_weights()[0])
np.save("tensorflow_topdown_directions/conv_biases", model.layers[0].get_weights()[1])

# print("Hidden Layer")
np.save("tensorflow_topdown_directions/hidden_weights", model.layers[3].get_weights()[0])
np.save("tensorflow_topdown_directions/hidden_biases", model.layers[3].get_weights()[1])
print(np.array(model.layers[3].get_weights()[0]).shape)

# print("Output Layer")
np.save("tensorflow_topdown_directions/output_weights", model.layers[4].get_weights()[0])
np.save("tensorflow_topdown_directions/output_biases", model.layers[4].get_weights()[1])


# predict using webcam
cap = cv2.VideoCapture(1)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        continue
    cv2.imshow('Input camera', image)
    # do some preprocessing before prediction
    loaded_image_new = np.array(loadImage(image)).reshape(1,input_shape[0],input_shape[1],input_shape[2])
    # print("loaded_image_new.shape: ", loaded_image_new.shape)
    cv2.imshow('Extracted Hand', loaded_image_new[0])
    new_prediction = model.predict(loaded_image_new, verbose=0)
    index_min = np.argmax(new_prediction[0])
    if(index_min == 0):
        print("Prediction: Up")
    if(index_min == 1):
        print("Prediction: Left")
    if(index_min == 2):
        print("Prediction: Right")
    if cv2.waitKey(5) == ord(" "):
        break

cap.release()

