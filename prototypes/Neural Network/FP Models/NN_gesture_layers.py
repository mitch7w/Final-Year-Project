# adapted to work with greyscale input (1 input channel)

import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import scipy.signal
import sys
from math import sqrt
from numpy.random import default_rng
import cv2
import random

# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

# set random seed for random number generator
rng = default_rng(777)

# set start time of program execution
start_time = time.time()

#open file to write out hyperparamaters and layer values to
details_file = open("network_details_gesture_layers.txt", "w+")
enableTextFileLogging = False
loadWeightsFromFile = False

# accepts arrays and strings and writes them in the network_details file
def writeToFile(input):
    if(enableTextFileLogging):
        details_file.write(str(input)+"\n\n")

# Load image, remove alpha channel and normalize input - change this method as needed for specific input data
def loadImage(imageName):
    if(isinstance(imageName,str)):
        # load image and remove alpha channel
        greyscale = cv2.imread(imageName,0) 
    else:
        greyscale = cv2.cvtColor(imageName, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(greyscale, (20,15), interpolation = cv2.INTER_AREA)
    # modified edge detection filter
    filter = np.array([[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]])
    output_conv = scipy.signal.convolve2d(resized, filter, 'valid')
    loadedImage = np.array(output_conv)/255 # divide by 255 in order to normalize input
    
    new3DImage = np.empty((13,18,1)) # convert greyscale to array
    for x in range(len(new3DImage)):
        for y in range(len(new3DImage[x])):
            for i in range(len(new3DImage[x][y])):
                new3DImage[x][y][i] = loadedImage[x][y]
    return np.array(new3DImage)

# Different layers available for use in neural network
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
        writeToFile("input_layer: " + str(self.output_neurons))

class Convolution:
    output_neurons = []
    filters = []
    delta = []
    
    def __init__(self, number_filters, kernel_size, activation_function):
        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        # initialize filters with random values from -1 to 1
        self.filters = rng.random(number_filters*kernel_size[0]*kernel_size[1]*kernel_size[2])
        for f in range(len(self.filters)):
            self.filters[f] = (self.filters[f] *2) -1  # generate new filter value -1 to 1
        self.filters=self.filters.reshape(number_filters,kernel_size[0],kernel_size[1],kernel_size[2])

    def get_size(self):
            return self.number_neurons

    def activation(self, x, activation_function):
        if(activation_function=="relu"):
            if(x>0):
                return x
            else:
                return 0
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
        
    def activateMatrix(self,matrix):
        # activate each value of a matrix and then return it
        matrix = np.array(matrix)
        original_shape = matrix.shape
        matrix=matrix.flatten()
        for i in range(len(matrix)):
            matrix[i] = self.activation(matrix[i], self.activation_function)
        return(matrix.reshape(original_shape))
    # convolution operation
    def convolve(self,input, filter):
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
        newOutput = scipy.signal.convolve2d(input,filter, "valid","fill",0) # temp replacement with scipy version
        return newOutput

    def fullConvolution(self,input,filter): # replaced with scipy.signal.convolve2d(..."full") for now
        # # Can use regular convolution method so long as right amount padding added
        # filterLength = filter.shape[0]
        # paddedInputLength = (3*filterLength) -2
        # # create new version of input with all padding
        # newInput = np.zeros(paddedInputLength**2).reshape((paddedInputLength,paddedInputLength))
        # xCounter = filterLength-1
        # yCounter = filterLength-1
        # for xCounter in range(filterLength-1,filterLength+3):
        #     for yCounter in range(filterLength-1,filterLength+3):
        #         newInput[xCounter][yCounter] = input[xCounter][yCounter] # put in input values that aren't zero
        # print(newInput)
        return scipy.signal.convolve2d(input,filter, "full", "fill",0)

    def forward_propagation(self, previous_layer_output):
        output_size = ( (len(previous_layer_output)-self.kernel_size[0]+1) ,(len(previous_layer_output[0])-self.kernel_size[1]+1) , self.number_filters )
        self.output_neurons = np.empty(output_size)
        # convolve each RGB channel with the relevant filter channel
        for f in range(self.number_filters): # for each filter
            # convolution_outputs = np.empty([self.number_filters, len(previous_layer_output[0])-len(self.filters[0])+1 , len(previous_layer_output[0])-len(self.filters[0])+1 ])
            convolution_outputs = []
            for num_channel in range(len(previous_layer_output[0][0])): # for each channel in previous layer
                convolution_outputs.append(self.convolve(previous_layer_output[:,:,num_channel], self.filters[f][:,:,num_channel])) # convolve each filter channel with each input image channel
            # sum the results of each R,G,B convolution for the output neuron
            for x in range(convolution_outputs[0].shape[0]):
                for y in range(convolution_outputs[0].shape[1]):
                    summedValue = 0
                    for i in range(len(convolution_outputs)):
                        summedValue += convolution_outputs[i][x][y] # sum each rgb channel
                    self.output_neurons[x][y][f] = summedValue # place each summed value in correct spot in output array
        self.output_neurons = self.activateMatrix(self.output_neurons) 
        self.output_neurons = np.array(self.output_neurons)
        # print("Shape at output of conv. layer 1: ", self.output_neurons.shape)
        writeToFile("convolution_output: " + str(self.output_neurons))
        
    def backpropagation(self, next_layer_delta, xy_max_pairs, input_data,learning_rate):
        convolution_gradient = np.zeros(self.output_neurons.shape[0]*self.output_neurons.shape[1]*self.number_filters).reshape((self.output_neurons.shape[0],self.output_neurons.shape[1],self.number_filters))
        # for every max-pooled value set its gradient
        next_layer_delta = np.array(next_layer_delta).flatten()
        next_layer_delta_counter = 0
        for num_filter in range(self.number_filters):
            for x in range(len(xy_max_pairs)):
                for y in range(len(xy_max_pairs[x])): # this loops to 704 but delta only 192 long
                    convolution_gradient[int(xy_max_pairs[x][y][num_filter][0])][int(xy_max_pairs[x][y][num_filter][1])][num_filter] = next_layer_delta[next_layer_delta_counter]
                    next_layer_delta_counter +=1
            # find filter gradients and update filter values for each R,G,B filter
        filter_gradients = np.empty(self.filters.shape)
        for num_filter in range(self.number_filters):
            channel_gradients = np.empty((self.filters.shape[1],self.filters.shape[2],self.filters.shape[3]))
            for channel_num in range(len(input_data[0][0])):
                channel_gradients[:,:,channel_num] = self.convolve(input_data[:,:,channel_num],convolution_gradient[:,:,num_filter])
            filter_gradients[num_filter] = channel_gradients
        # update filter values with F -= learningRate *dL/dF
        for num_filter in range(len(self.filters)):
            for channel_num in range(len(input_data[0][0])):
                for x in range(len(self.filters[num_filter])):
                    for y in range(len(self.filters[num_filter][x])):
                        self.filters[num_filter][x][y][channel_num] -= learning_rate * filter_gradients[num_filter][x][y][channel_num]
        # update input matrix X with dL/dX = Full convolvution of 180deg rotated Filter F and loss gradient dL/dO
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
        writeToFile("Filters: " + str(self.filters))

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

    # TODO implement striding
    def forward_propagation(self, previous_layer_output):
        self.xy_max_pairs = np.empty((previous_layer_output.shape[0]-self.pool_size[0]+1,previous_layer_output.shape[1]-self.pool_size[1]+1, previous_layer_output.shape[2], 2))
        self.output_neurons = np.empty((previous_layer_output.shape[0]-self.pool_size[0]+1,previous_layer_output.shape[1]-self.pool_size[1]+1, previous_layer_output.shape[2])) # new dimensions = previous dimensions -2
        for filter_num in range(len(previous_layer_output[0][0])):
            # dimensions of previous layer
            previousLayer_x_size = previous_layer_output.shape[0]
            previousLayer_y_size = previous_layer_output.shape[1]
            # inputX and inputY are counters over input layer
            inputX = 0
            inputY = 0
            # iterate over the entire input matrix
            while(inputX+self.pool_size[0] < previousLayer_x_size+1):
                while(inputY+self.pool_size[1] < previousLayer_y_size+1):
                    # filterX and filterY are the x and y positions going across the length of the imaginary pooling matrix
                    # iterate over the whole imaginary pooling matrix
                    maxValue = -sys.maxsize
                    x_max = -1
                    y_max = -1
                    for filterX in range(0,self.pool_size[0]):
                        for filterY in range(0,self.pool_size[1]):
                            # find the max value from the current pool and set the releveant element of pool matrix to it
                            if( previous_layer_output[inputX+filterX][inputY+filterY][filter_num] > maxValue ):
                                maxValue = previous_layer_output[inputX+filterX][inputY+filterY][filter_num]
                                x_max = inputX + filterX
                                y_max = inputY + filterY
                    self.output_neurons[inputX][inputY][filter_num] = maxValue # set each element of pool matrix = max value
                    self.xy_max_pairs[inputX][inputY][filter_num] = (x_max,y_max) # store where max element was so can find it easily in backpropogation
                    inputY +=1
                inputX +=1
                inputY = 0
        writeToFile("Maxpooling layer: " + str(self.output_neurons))

    def backpropagation(self,next_layer_delta):
        self.delta = next_layer_delta

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
        self.output_neurons = np.array(previous_layer_output).flatten()
        writeToFile("Flatten layer: " + str(self.output_neurons))

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
        self.weights = rng.random(previous_layer_output_size*self.number_neurons)
        for w in range(len(self.weights)):
            self.weights[w] = (self.weights[w] *0.2) -0.1
        self.weights=self.weights.reshape(self.number_neurons,previous_layer_output_size)
        # self.bias = rng.random(self.number_neurons)
        # He method
        # self.weights = rng.random(previous_layer_output_size*self.number_neurons)
        # self.bias = rng.random(self.number_neurons)
        self.bias = [0.1] * self.number_neurons
        # standard_deviation = sqrt(2/previous_layer_output_size)
        # for w in range(len(self.weights)):
        #     self.weights[w] = self.weights[w] * standard_deviation
        # self.weights=self.weights.reshape(self.number_neurons,previous_layer_output_size)

    def get_output(self):
        return self.output_neurons

    def activation(self, x, activation_function):
        if(activation_function=="relu"):
            if(x>0):
                return x
            else:
                return 0
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
        self.output_neurons=[]
        for neuron_number in range(self.number_neurons):
            # update hidden layer with dot product of inputs * weights all run through the activation function
            activationInput = np.dot(previous_layer_output , self.weights[neuron_number])+ self.bias[neuron_number]
            self.output_neurons.append(self.activation(activationInput, self.activation_function))
        writeToFile("Hidden layer: " + str(self.output_neurons))

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
        # update weights hidden weights
        for i in range(len(self.weights)): # self.weights[i] = each hidden neuron
            for j in range(len(self.weights[i])): # self.weights[i][j] = hidden neuron's connection to each previous layer jth neuron
                self.weights[i][j] -= learning_rate * self.hidden_delta[i] * previous_layer_output[j]
            # update hidden biases
            self.bias[i] -= learning_rate * self.hidden_delta[i]
        # writeToFile("hidden_layer_weights: " + str(self.weights))

    def get_weights(self):
        return self.weights

    def get_delta(self):
        return self.hidden_delta
    
    def get_biases(self):
        return self.bias

class Output:
    # Only difference from hidden is how backpropagation is conducted
    output_neurons = []
    output_delta = []

    def __init__(self, number_neurons, activation_function):
        self.number_neurons = number_neurons
        self.activation_function = activation_function
    
    def get_size(self):
        return self.number_neurons

    # initialize weights and biases with the He method
    def initialize_He_weights(self,previous_layer_output_size):
        self.weights = rng.random(previous_layer_output_size*self.number_neurons)
        for w in range(len(self.weights)):
            self.weights[w] = (self.weights[w] *0.2) -0.1
        self.weights=self.weights.reshape(self.number_neurons,previous_layer_output_size)
        # self.bias = rng.random(self.number_neurons)
        # He method
        # self.weights = rng.random(previous_layer_output_size*self.number_neurons)
        # self.bias = rng.random(self.number_neurons)
        self.bias = [0.1] * self.number_neurons
        # standard_deviation = sqrt(2/previous_layer_output_size)
        # for w in range(len(self.weights)):
        #     self.weights[w] = self.weights[w] * standard_deviation
        # self.weights=self.weights.reshape(self.number_neurons,previous_layer_output_size)

    def get_output(self):
        return self.output_neurons

    def activation(self, x, activation_function):
        if(activation_function=="relu"):
            if(x>0):
                return x
            else:
                return 0
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
        self.output_neurons=[]
        for neuron_number in range(self.number_neurons):
            # update hidden layer with dot product of inputs * weights all run through the activation function
            activationInput = np.dot(previous_layer_output , self.weights[neuron_number])+ self.bias[neuron_number]
            self.output_neurons.append(self.activation(activationInput, self.activation_function))
        writeToFile("Output layer: " + str(self.output_neurons))

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
    networkInitialized = False

    def __init__(self, layers):
        self.layers = layers

    # perform forward propogation of whole network based on new input data
    def forward_propagation(self, new_input_data):
        for i in range(len(self.layers)):
            if(i==0):
                self.layers[i].forward_propagation(new_input_data) # pass in the new input data
            else:
                # Initialize weights if first time through network
                if(self.networkInitialized == False):
                    if(isinstance(self.layers[i], Hidden) or isinstance(self.layers[i], Output)):
                        self.layers[i].initialize_He_weights(self.layers[i-1].get_size()) # initialize weights with He method based on size of previous layer
                # propagate forwards with previous layer's output as input to next layer
                self.layers[i].forward_propagation(self.layers[i-1].get_output())
        self.networkInitialized = True

    def predict(self, new_input_data):
        self.forward_propagation(new_input_data)
        return self.layers[len(self.layers)-1].get_output()

    def back_propagation(self, input_data, desired_output,learning_rate):
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
    
    
    def train(self, training_data_input, training_data_output, learning_rate, num_training_epochs):
        writeToFile("learning_rate: " + str(learning_rate))
        writeToFile("num_training_epochs: " + str(num_training_epochs))
        for epoch in range(num_training_epochs):
            epoch_start_time = time.time()
            total_error_at_output = 0
            # for each set of training data
            length_training_data = len(training_data_input)
            for training_data_num in range(length_training_data):
                self.forward_propagation(training_data_input[training_data_num])
                total_error_at_output += sum( (np.array(training_data_output[training_data_num]) -np.array(self.layers[len(self.layers)-1].get_output()))**2)
                self.back_propagation(training_data_input[training_data_num], training_data_output[training_data_num],learning_rate)        
                currentProgressString = str(training_data_num) + "/", str(length_training_data)
                print(training_data_num, "/", length_training_data, end='\r')
            epoch_end_time = time.time()
            print("Epoch: ", epoch, end=", ")
            print("Learning rate: ", learning_rate, end=", ")
            print("Error: ", total_error_at_output, end=", ")
            print(f"Runtime: {(epoch_end_time - epoch_start_time):.2f} seconds")
            self.epochErrors.append(total_error_at_output)
            if(total_error_at_output < 1):
                np.save('Trained Nets/NN_gesture_layers_hidden_weights', model.layers[2].get_weights())
                np.save('Trained Nets/NN_gesture_layers_output_weights', model.layers[3].get_weights())
                np.save('Trained Nets/NN_gesture_layers_hidden_bias', model.layers[2].get_biases())
                np.save('Trained Nets/NN_gesture_layers_output_bias', model.layers[3].get_biases())
        plt.plot(range(len(self.epochErrors)), self.epochErrors)
        plt.xlabel("Num Epochs")
        plt.ylabel("Training Square Loss")
        plt.show()

        # Overfitting dataset checks
        for i in range(len(training_data_input)):
            prediction = self.predict(training_data_input[i])
            print("Expected: ", training_data_output[i], ", Got: ", prediction)

### load training data ###
training_data_input = []
for i in range(240):
    imageName = "gesturetrainingdata/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImage(imageName))
# each gesture0.npy has [x,y,z] with each x, y and z being [thumbx, pinkyx...] from 0-1. Ignore z values
training_data_output = []
for i in range(120):
    training_data_output.append([1,0]) # open hand
for i in range(120):
    training_data_output.append([0,1]) # fist

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

### set up neural network ###
model = Network([
    Input("sigmoid"),
    Convolution(20, (3,3,1), "sigmoid"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(2000, "sigmoid"),
    Output(2, "sigmoid")])
    # Deep pose has poolsizes decreasing and number filters increasing for subsequent conv. layers
### train network ###
# load weights from pre-trained NN if so desired
if(loadWeightsFromFile):
    model.layers[1].filters = np.load('Trained Nets/NN_gesture_layers_conv_layer_filter.npy')
    model.layers[4].weights = np.load('Trained Nets/NN_gesture_layers_hidden_weights.npy')
    model.layers[5].weights = np.load('Trained Nets/NN_gesture_layers_output_weights.npy')
    model.layers[4].bias = np.load('Trained Nets/NN_gesture_layers_hidden_bias.npy')
    model.layers[5].bias = np.load('Trained Nets/NN_gesture_layers_output_bias.npy')
    model.networkInitialized = True
# model.train(training_data_input , training_data_output, learning_rate=0.01, num_training_epochs=8000)

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
    loaded_image_new = loadImage(image)
    new_prediction = model.predict(loaded_image_new)
    if(new_prediction[0]>new_prediction[1]): # softmax layer basically
        print("Prediction: Open Hand")
    else:
        print("Prediction: Fist")
    if cv2.waitKey(5) == ord(" "):
        break

cap.release()


details_file.close() # close network details textfile

# end program runtime and report duration
end_time = time.time() 
print(f"Program runtime: {(end_time - start_time):.2f} seconds")