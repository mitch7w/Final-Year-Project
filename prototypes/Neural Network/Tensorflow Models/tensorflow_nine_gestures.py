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
    # while True:
    #     cv2.imshow("Check Output", result.astype(np.uint8))
    #     if cv2.waitKey(5) == ord(" "):
    #         break
    greyscale = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(mask, (320,240), interpolation = cv2.INTER_AREA)
    # modified edge detection filter
    filter = np.array([[0, -1, 0],
                    [-1, 4, -1],
                    [0, -1, 0]])
    output_conv = scipy.signal.convolve2d(resized, filter, 'valid')
    # load in already segmented image
    loadedImage = np.array(output_conv)/255 # divide by 255 in order to normalize input
    new3DImage = np.empty((loadedImage.shape[0],loadedImage.shape[1],1)) # convert greyscale to array
    for x in range(len(new3DImage)):
        for y in range(len(new3DImage[x])):
            for i in range(len(new3DImage[x][y])):
                new3DImage[x][y][i] = loadedImage[x][y]
    return np.array(new3DImage)



training_data_input = []

# no faces input
# for i in range(431):
#     print(i,end="")
#     imageName = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Neural network/nine_gestures_no_faces/gesture" + str(i) + ".jpg"
#     training_data_input.append(loadImage(imageName))

for i in range(1909):
    imageName = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Neural network/nine_gestures/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImage(imageName))
    
for i in range(2333,4242):
    imageName = "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Neural network/nine_gestures/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImage(imageName))

training_data_output = []
for j in range(2):
  for i in range(212):
      training_data_output.append([1,0,0,0,0,0,0,0,0]) # one
  for i in range(212):
      training_data_output.append([0,1,0,0,0,0,0,0,0]) # two
  for i in range(212):
      training_data_output.append([0,0,1,0,0,0,0,0,0]) # three
  for i in range(212):
      training_data_output.append([0,0,0,1,0,0,0,0,0]) # four
  for i in range(212):
      training_data_output.append([0,0,0,0,1,0,0,0,0]) # five
  for i in range(212):
      training_data_output.append([0,0,0,0,0,1,0,0,0]) # fist
  for i in range(212):
      training_data_output.append([0,0,0,0,0,0,1,0,0]) # peace
  for i in range(212):
      training_data_output.append([0,0,0,0,0,0,0,1,0]) # rockon
  for i in range(212):
      training_data_output.append([0,0,0,0,0,0,0,0,1]) # okay
      

# no faces output
# training_data_output = []
# for j in range(2):
#   for i in range(25):
#       training_data_output.append([1,0,0,0,0,0,0,0,0]) # one
#   for i in range(25):
#       training_data_output.append([0,1,0,0,0,0,0,0,0]) # two
#   for i in range(25):
#       training_data_output.append([0,0,1,0,0,0,0,0,0]) # three
#   for i in range(25):
#       training_data_output.append([0,0,0,1,0,0,0,0,0]) # four
#   for i in range(25):
#       training_data_output.append([0,0,0,0,1,0,0,0,0]) # five
#   for i in range(25):
#       training_data_output.append([0,0,0,0,0,1,0,0,0]) # fist
#   for i in range(25):
#       training_data_output.append([0,0,0,0,0,0,1,0,0]) # peace
#   for i in range(25):
#       training_data_output.append([0,0,0,0,0,0,0,1,0]) # rockon
#   for i in range(25):
#       training_data_output.append([0,0,0,0,0,0,0,0,1]) # okay

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data

x_train = np.array(training_data_input[:-300])
y_train = np.array(training_data_output[:-300])
x_test = np.array(training_data_input[-300:])
y_test = np.array(training_data_output[-300:])
print("x_train.shape: ", x_train.shape)

num_output_classes = 9
input_shape = training_data_input[0].shape
print(input_shape)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(8, kernel_size=(4,4), activation="relu"),
        layers.MaxPooling2D(pool_size=(8,8)),
        layers.Conv2D(16, kernel_size=(2, 2), activation="relu"),
        layers.MaxPooling2D(pool_size=(4,4),strides = (4,4)),
        layers.Conv2D(16, kernel_size=(2, 2), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2),strides = (2,2)),
        layers.Flatten(),
        layers.Dense(100, activation="relu"),
        layers.Dense(num_output_classes, activation="softmax"),
    ]
)

model.summary()

#Train the model
batch_size = 1
epochs = 40

# Setup model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# # Early stopping
# callback = callbacks.EarlyStopping(monitor='accuracy', patience=1)
# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[callback],verbose=1)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=1)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# predict using webcam
cap = cv2.VideoCapture(0)
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
        print("Prediction: Five")
    if(index_min==1):
        print("Prediction: Fist")
    if(index_min==2):
        print("Prediction: Peace")
    if cv2.waitKey(5) == ord(" "):
        break

cap.release()


