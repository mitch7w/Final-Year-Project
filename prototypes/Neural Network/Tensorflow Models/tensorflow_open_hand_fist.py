# -*- coding: utf-8 -*-

# Setup
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import scipy.signal
import cv2
from numpy.random import default_rng
import random
# set random seed for random number generator
rng = default_rng(777)

def loadImage(imageName):
    if(isinstance(imageName,str)):
        # load image if just it's name was passed in
        imageName = cv2.imread(imageName,1) 
    # imageName is now the RGB-passed-in image
    # perform skin segmentation using the Ycbcr color space
    image_ycrcb = np.array(cv2.cvtColor(imageName, cv2.COLOR_RGB2YCR_CB))
    skin1 = (50, 89, 136)
    skin2 = (231, 147, 181)
    mask = cv2.inRange(image_ycrcb, skin1, skin2)
    result = cv2.bitwise_and(imageName, imageName, mask=mask)
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
    # extract hand
    loadedImage = np.array(result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]])/255 # divide by 255 in order to normalize input
    new3DImage = np.empty((loadedImage.shape[0],loadedImage.shape[1],1)) # convert greyscale to array
    for x in range(len(new3DImage)):
        for y in range(len(new3DImage[x])):
            for i in range(len(new3DImage[x][y])):
                new3DImage[x][y][i] = loadedImage[x][y]
    return np.array(new3DImage)

training_data_input = []
for i in range(120):
    imageName = "fist_classifier_new_webcam/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImage(imageName))
training_data_output = []
for i in range(40):
    training_data_output.append([1,0]) # Open
for i in range(40):
    training_data_output.append([0,1]) # Fist

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data

x_train = np.array(training_data_input[:-12])
y_train = np.array(training_data_output[:-12])
x_test = np.array(training_data_input[-12:])
y_test = np.array(training_data_output[-12:])
print("x_train.shape: ", x_train.shape)

num_output_classes = 4
input_shape = training_data_input[0].shape
print(input_shape)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(96, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        # layers.Conv2D(192, kernel_size=(3, 3), activation="relu"),
        # layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(num_output_classes, activation="softmax"),
    ]
)

model.summary()

#Train the model
batch_size = 1
epochs = 5

# Setup model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Stop the training if accuracy of 1 is reached
callback = callbacks.EarlyStopping(monitor='accuracy', patience=1)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

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
    cv2.imshow('Input Gesture', cv2.flip(image, 1))
    # do some preprocessing before prediction
    loaded_image_new = np.array(loadImage(image)).reshape(1,input_shape[0],input_shape[1],input_shape[2])
    print("loaded_image_new: ", loaded_image_new.shape)
    new_prediction = model.predict(loaded_image_new)
    print("new_prediction: ", new_prediction,end="")
    index_min = np.argmax(new_prediction[0])
    if(index_min==0):
        print("Prediction: Empty")
    if(index_min==1):
        print("Prediction: Face")
    if(index_min==2):
        print("Prediction: Open")
    if(index_min==3):
        print("Prediction: Fist")
    if cv2.waitKey(5) == ord(" "):
        break

cap.release()


