import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import time
import math
import collections
from PIL import Image
import scipy.signal
import sys
from math import sqrt
from numpy.random import randn
np.set_printoptions(threshold=sys.maxsize) # makes arrays print out in full

start_time = time.time()

#open file to write out hyperparamaters and layer values to
details_file = open("network_details_mnist.txt", "w+")
enableTextFileLogging = True

# accepts arrays and strings and writes them in the network_details file
def writeToFile(input):
    if(enableTextFileLogging):
        details_file.write(str(input)+"\n\n")

# random seed setting
rng = np.random.default_rng(1)

# replace with scipy.signal.convolve2d(..."full") for now
# def fullConvolution(input,filter):
#     # Can use regular convolution method so long as right amount padding added
#     filterLength = filter.shape[0]
#     paddedInputLength = (3*filterLength) -2
#     # create new version of input with all padding
#     newInput = np.zeros(paddedInputLength**2).reshape((paddedInputLength,paddedInputLength))
#     xCounter = filterLength-1
#     yCounter = filterLength-1
#     for xCounter in range(filterLength-1,filterLength+3):
#         for yCounter in range(filterLength-1,filterLength+3):
#             newInput[xCounter][yCounter] = input[xCounter][yCounter] # put in input values that aren't zero
#     print(newInput)

def maxPooling(previousLayer):
    # dimensions of previous layer
    previousLayer_x_size = previousLayer.shape[0]
    previousLayer_y_size = previousLayer.shape[1]
    poolLength = 3 # 3x3 max pooling layer
    strideLength = 2
    pooledLayer = np.empty((previousLayer.shape[0]-poolLength+1, previousLayer.shape[1]-poolLength+1)) # new dimensions = previous dimensions -2
    # inputX and inputY are counters over input layer
    inputX = 0
    inputY = 0
    # iterate over the entire input matrix
    while(inputX+poolLength < previousLayer_x_size+1):
        while(inputY+poolLength < previousLayer_y_size+1):
            # filterX and filterY are the x and y positions going across the length of the imaginary pooling matrix
            # iterate over the whole imaginary pooling matrix
            maxValue = -1
            for filterX in range(0,poolLength):
                for filterY in range(0,poolLength):
                    # find the max value from the current pool and set the releveant element of pool matrix to it
                    if( previousLayer[inputX+filterX][inputY+filterY] > maxValue ):
                        maxValue = previousLayer[inputX+filterX][inputY+filterY]
            pooledLayer[inputX][inputY] = maxValue # set each element of pool matrix = max value
            inputY +=1
        inputX +=1
        inputY = 0
    return pooledLayer

# temporarily replaced with scipy method to check speed
def convolution(input, filter):
    # # performs convolution with an input matrix and filter matrix
    # # filter and input matrix dimensions
    # filter_x_size = filter.shape[0]
    # filter_y_size = filter.shape[1]
    # input_x_size = input.shape[0]
    # input_y_size = input.shape[1]
    # # set output matrix size - no strides or padding
    # newOutput = np.empty([input_x_size-filter_x_size+1, input_y_size-filter_y_size+1],dtype=np.int64) 
    # # inputX and inputY are counters specifying where the top left corner of the currently considered input matrix is
    # inputX = 0
    # inputY = 0
    # # iterate over the entire input matrix
    # while(inputX+filter_x_size < input_x_size+1):
    #     while(inputY+filter_y_size < input_y_size+1):
    #         # filterX and filterY are the x and y positions going across the filter matrix
    #         # iterate over the whole filter matrix
    #         sumMultiplications = 0.0 # add all the multiplications together
    #         for filterX in range(0,filter_x_size):
    #             for filterY in range(0,filter_y_size):
    #                 print("Multiplying input", input[inputX+filterX][inputY+filterY], "with filter", filter[filterX][filterY])
    #                 # each element of current subsection of input matrix * each element of filter matrix
    #                 sumMultiplications += input[inputX+filterX][inputY+filterY] * filter[filterX][filterY]
    #         newOutput[inputX][inputY] = sumMultiplications # set each element of output matrix = sum of multiplications
    #         inputY +=1
    #     inputX +=1
    #     inputY = 0
    newOutput = scipy.signal.convolve2d(input,filter, "valid","fill",0)
    return newOutput

def loadImage(imageName):
    # load image and remove alpha channel
    with Image.open(imageName) as im:
        loadedImage = np.array(im)/255 # divide by 255 in order to normalize input
        new3DImage = np.empty((28,28,3))
        for x in range(len(new3DImage)):
            for y in range(len(new3DImage[x])):
                new3DImage[x][y] = [loadedImage[x][y],loadedImage[x][y],loadedImage[x][y]]
        return new3DImage

def activation(x):
    return 1/(1+np.e**(-x))
    # if(x>0):
    #     return x
    # else:
    #     return 0

def activation_derivative(x):
    z= 1/(1+np.e**(-x))
    return z * (1-z)
    # if(x>0):
    #     return 1
    # else:
    #     return 0

def forward_propagation(currentImage):
    global convolution_output, input_layer, hidden_layer, output_layer, hidden_bias, output_bias, weights_1, weights_2, filter1   
    
    # print("Starting forward prop.")

    # convolutional layer
    convolution_output_red = convolution(currentImage[:,:,0], filter1[0])
    convolution_output_green = convolution(currentImage[:,:,1], filter1[1])
    convolution_output_blue = convolution(currentImage[:,:,2], filter1[2])
    # writeToFile("currentImage[:,:,0]: " + str(currentImage[:,:,0]))
    # writeToFile("filter1[0]: " + str(filter1[0]))
    # writeToFile("convolution_output_red: " + str(convolution_output_red))
    
    # sum the results of each R,G,B convolution for the output convolution
    convolution_output = np.empty((convolution_output_red.shape[0], convolution_output_red.shape[1]))
    for x in range(convolution_output.shape[0]):
        for y in range(convolution_output.shape[1]):
            convolution_output[x][y] = convolution_output_red[x][y] + convolution_output_green[x][y] + convolution_output_blue[x][y]        
            # convolution_output[x][y] = convolution_output[x][y] / 765 # maybe neurons have to be 0-1
            # maybe add bias to convolutional adding
    writeToFile("convolution_output: " + str(convolution_output))

    # max pooling layer
    max_pooled1 = maxPooling(convolution_output)
    max_pooled1_flattened = max_pooled1.flatten() # make max pooling layer into 1D array for input into fully-connected NN
    #newInputLayer = output of max pooling layer 1
    input_layer = max_pooled1_flattened
    writeToFile("input_layer: " + str(input_layer))
    for neuron_number in range(len(hidden_layer)):
        # update hidden layer with dot product of inputs * weights all run through the activation function
        activationInput = np.dot(input_layer , weights_1[neuron_number])+ hidden_bias[neuron_number]
        hidden_layer[neuron_number] = activation(activationInput)
    writeToFile("hidden_layer:" + str(hidden_layer))
    for neuron_number in range(len(output_layer)):
        # update output layer with dot product of hidden layer * weights all run through the activation function
        output_layer[neuron_number] = activation(np.dot(hidden_layer , weights_2[neuron_number]) + output_bias[neuron_number])
    writeToFile("output_layer: " + str(output_layer))
    # print("Ended forward prop.")

def predict(inValues):
    global output_layer
    forward_propagation(inValues)
    return output_layer

def back_propagation(currentInputImage,desired_output):
    global convolution_output, input_layer, hidden_layer, output_layer, hidden_bias, output_bias, weights_1, weights_2, learning_rate, filter1   
    
    # print("Starting backward prop.")
    
    # calculate errors for output layer
    output_errors = output_layer - desired_output
    # calculate delta of output layer - how much they must change
    output_delta = []
    for i in range(len(output_errors)):
        output_delta.append(output_errors[i] * activation_derivative(output_layer[i]))
    # calculate errors for hidden layer
    # for each neuron in hidden layer must have a error of its weights * neuron in next layer's delta
    hidden_errors = [0] * len(hidden_layer)
    for i in range(len(weights_2)): # weights_2[i] = each output neuron's weight array
        for j in range(len(weights_2[i])): # weights_2[i][j] = output neuron's connection to each hidden layer jth neuron
            # for each neuron in outputlayer, add to error the weight*output delta
            hidden_errors[j] += weights_2[i][j] * output_delta[i]
    hidden_delta = []
    for i in range(len(hidden_errors)):
        hidden_delta.append(hidden_errors[i] * activation_derivative(hidden_layer[i]))
    # calculate errors for input layer
    input_errors = [0] * len(input_layer)
    for i in range(len(weights_1)): # weights_1[i] = each hidden neuron
        for j in range(len(weights_1[i])): # weights_1[i][j] = hidden neuron's connection to each input layer neuron
            input_errors[j] += weights_1[i][j] * hidden_delta[i]
    input_delta = []
    for i in range(len(input_errors)):
        input_delta.append(input_errors[i] * activation_derivative(input_layer[i]))
    # update fully-connected NN components
    
    # update weights from input to hidden
    for i in range(len(weights_1)): # weights_1[i] = each hidden neuron
        for j in range(len(weights_1[i])): # weights_1[i][j] = hidden neuron's connection to each input layer neuron
            weights_1[i][j] -= learning_rate * hidden_delta[i] * input_layer[j]
        # update hidden biases
        hidden_bias[i] -= learning_rate * hidden_delta[i]
    # writeToFile("weights_1: " + str(weights_1))
    # update weights from hidden to output
    for i in range(len(weights_2)): # weights_2[i] = each output neuron
        for j in range(len(weights_2[i])): # weights_2[i][j] = output neuron's connection to each hidden layer neuron
            weights_2[i][j] -= learning_rate * output_delta[i] * hidden_layer[j]
        # update output biases
        output_bias[i] -= learning_rate * output_delta[i]  
    # writeToFile("weights_2: " + str(weights_2))
    # update convolutional components

    # TODO replace max pooling search for max values with array set during forward prop.
    # update max-pooling layers
    convolution_gradient = np.zeros(convolution_output.shape[0]**2).reshape((convolution_output.shape[0],convolution_output.shape[1]))
    for i in range(len(input_layer)): # for every max pool value find it in conv layer and set its gradient
        for x in range(len(convolution_output)):
            for y in range(len(convolution_output[x])):
                if(convolution_output[x][y] == input_layer[i]):
                    convolution_gradient[x][y] = input_delta[i] # set gradient for conv layer
                # don't need to modify input_layer (max_pooled output) as no learnable parameters here
    
    # find filter gradients and update filter values for each R,G,B filter - filter1[0] to filter1[2]
    filter1_gradient0 = convolution(currentInputImage[:,:,0],convolution_gradient)
    filter1_gradient1 = convolution(currentInputImage[:,:,1],convolution_gradient)
    filter1_gradient2 = convolution(currentInputImage[:,:,2],convolution_gradient)
    # update filter values with F -= learningRate *dL/dF
    for x in range(len(filter1_gradient0)):
        for y in range(len(filter1_gradient0[x])):
            filter1[0][x][y] -= learning_rate * filter1_gradient0[x][y] 
            filter1[1][x][y] -= learning_rate * filter1_gradient1[x][y] 
            filter1[2][x][y] -= learning_rate * filter1_gradient2[x][y] 
    writeToFile("filter1: " + str(filter1))
    # update input matrix X with dL/dX = Full convolvution of 180deg rotated Filter F and loss gradient dL/dO
    rotatedFilter0 = []
    rotatedFilter1 = []
    rotatedFilter2 = []
    for x in range(len(filter1[0])):
        for y in range(len(filter1[0][x])):
            rotatedFilter0.append(filter1[0][x][y])
            rotatedFilter1.append(filter1[1][x][y])
            rotatedFilter2.append(filter1[2][x][y])
    # rotate filter about vertical then horizontal axis 
    rotatedFilter0.reverse()
    rotatedFilter0 = np.array(rotatedFilter0).reshape((filter1.shape[0],filter1.shape[1]))
    rotatedFilter1.reverse()
    rotatedFilter1 = np.array(rotatedFilter1).reshape((filter1.shape[0],filter1.shape[1]))
    rotatedFilter2.reverse()
    rotatedFilter2 = np.array(rotatedFilter2).reshape((filter1.shape[0],filter1.shape[1]))
    inputImageDelta0 = scipy.signal.convolve2d(rotatedFilter0,convolution_gradient, "full", "fill",0) # TODO replace Full convolution with FP implementation
    inputImageDelta1 = scipy.signal.convolve2d(rotatedFilter1,convolution_gradient, "full", "fill",0) 
    inputImageDelta2 = scipy.signal.convolve2d(rotatedFilter2,convolution_gradient, "full", "fill",0) 
    inputImageDelta = [] # gradient for previous layer if needed
    for i in range(len(inputImageDelta0)):
        inputImageDelta.append(inputImageDelta0[i]+inputImageDelta1[i]+inputImageDelta2[i])
        # if giving massive inputs or gradient changes at start then must average these 3 values out or something
    # print("Ended backward prop.")

def train(training_data_input, training_data_output):
    global input_layer, hidden_layer, output_layer, hidden_bias, output_bias, weights_1, weights_2
    epochErrors = []
    for epoch in range(num_training_epochs):
        epoch_start_time = time.time()
        total_error_at_output = 0
        # for each set of training data
        for training_data_num in range(len(training_data_input)):
            forward_propagation(training_data_input[training_data_num])
            # calculate total error at output after each epoch
            # print("Expected output layer: ", training_data_output[training_data_num])
            # print("Actual output layer: ", output_layer)
            total_error_at_output += sum( (training_data_output[training_data_num] -output_layer)**2)
            back_propagation(training_data_input[training_data_num], training_data_output[training_data_num])        
        epoch_end_time = time.time()
        print("Epoch: ", epoch, end=", ")
        print("Learning rate: ", learning_rate, end=", ")
        print("Error: ", total_error_at_output, end=", ")
        print(f"Runtime: {(epoch_end_time - epoch_start_time):.2f} seconds")
        epochErrors.append(total_error_at_output)
    plt.plot(range(len(epochErrors)), epochErrors)
    plt.xlabel("Num Epochs")
    plt.ylabel("Training Square Loss")
    plt.show()
        

############################################################################################################
############################################################################################################

# Inital parameter setup

############################################################################################################
############################################################################################################

learning_rate = 0.001
num_training_epochs = 200

writeToFile("learning_rate: " + str(learning_rate))
writeToFile("num_training_epochs: " + str(num_training_epochs))

# initialize network
######################################################
# inputImage = loadImage("musk_28x28.jpg") # import an image for initial input to network
training_data_input = [loadImage("0_28x28.png"),loadImage("1_28x28.png"),loadImage("2_28x28.png"),loadImage("3_28x28.png"),loadImage("4_28x28.png"),loadImage("5_28x28.png"),loadImage("6_28x28.png"),loadImage("7_28x28.png"),loadImage("8_28x28.png"),loadImage("9_28x28.png")]
training_data_output = [[1,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,1]]
# training_data_input = [loadImage("0_28x28.png")]
# training_data_output = [[1,0,0,0,0,0,0,0,0,0]]

# filters plus pooling layers
convolution_output = np.empty((0,0)) # output of convolutional layer of original image
filter1 = rng.random(27)
for f in range(len(filter1)):
    filter1[f] = (filter1[f] *2) -1
filter1 = filter1.reshape(3,3,3) # 3 filters of size 3x3 each for R,G,B
# writeToFile("filter1: " + str(filter1))
input_layer = np.empty(576)
hidden_layer = np.empty(1000)
output_layer = np.empty(10)
hidden_bias = rng.random(1000)
output_bias = rng.random(10)
# initialize weights with random values between -1 and 1
# He weight initialization
previous_layer_neurons_1 = 576
std_1 = sqrt(2/previous_layer_neurons_1)
previous_layer_neurons_2 = 1000
std_2 = sqrt(2/previous_layer_neurons_2)
weights_1 = randn(576*1000)
weights_2 = randn(1000*10)
for w in range(len(weights_1)):
    weights_1[w] = weights_1[w] *std_1
for w in range(len(weights_2)):
    weights_2[w] = weights_2[w] *std_2
weights_1=weights_1.reshape(1000,576) # array of weight inputs for hidden_layer (6x6) = [hidden_neuron1_weights ... hidden_neuron6_weights]
weights_2=weights_2.reshape(10,1000) # array of weight inputs for output_layer (6x6) = [output_neuron1_weights...output_neuron2_weights]
# Train NN on one image - overfit and check CNN actually learning
train(training_data_input, training_data_output)

# Overfitting dataset checks
for i in range(len(training_data_input)):
    prediction = predict(training_data_input[i])
    print("Expected: ", training_data_output[i], ", Got: ", prediction)

details_file.close() # close network details textfile

end_time = time.time()
print(f"Program runtime: {(end_time - start_time):.2f} seconds")
