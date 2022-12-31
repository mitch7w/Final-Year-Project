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

imageReadIn = cv2.imread("/Users/mitch/Documents/University/Project/Project GitLab/prototypes/topdown_view_sizing.jpg",1) 
image = cv2.resize(imageReadIn, (10,7), interpolation = cv2.INTER_AREA) # width x height
loadedImage = np.array(image)/255

training_data_input = []
for i in range(10):
    training_data_input.append(loadedImage)
training_data_output = []
for i in range(10):
    training_data_output.append([1,0]) # Open

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data

x_train = np.array(training_data_input)
y_train = np.array(training_data_output)
x_test = np.array(training_data_input)
y_test = np.array(training_data_output)
print("x_train.shape: ", x_train.shape)

num_output_classes = 2
input_shape = training_data_input[0].shape
print(input_shape)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(4, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
        layers.Flatten(),
        layers.Dense(5, activation="relu"),
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

print("Convolutional Layer")
np.save("tensorflow_small_example/conv_weights", model.layers[0].get_weights()[0])
np.save("tensorflow_small_example/conv_biases", model.layers[0].get_weights()[1])

# print("Hidden Layer")
np.save("tensorflow_small_example/hidden_weights", model.layers[3].get_weights()[0])
np.save("tensorflow_small_example/hidden_biases", model.layers[3].get_weights()[1])
print(np.array(model.layers[3].get_weights()[0]).shape)

# print("Output Layer")
np.save("tensorflow_small_example/output_weights", model.layers[4].get_weights()[0])
np.save("tensorflow_small_example/output_biases", model.layers[4].get_weights()[1])
print(loadedImage.shape)
print("get output at each layer for testing of FP NN")
details_file = open("tensorflow_small_example_outputs.txt", "w+")
# details_file.write(str(loadedImage)+"\n\n")
conv_out = model.layers[0](loadedImage.reshape((1,7,10,3))) # 1,height,width,channels
details_file.write(str(conv_out)+"\n\n")
max_out = model.layers[1](conv_out)
details_file.write(str(max_out)+"\n\n")
flatten = model.layers[2](max_out)
dense_out = model.layers[3](flatten)
out_out = model.layers[4](dense_out)
# details_file.write(str(dense_out)+"\n\n")
# details_file.write(str(out_out)+"\n\n")
exit()

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


