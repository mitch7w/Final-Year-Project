from urllib.parse import parse_qsl
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import scipy.signal
import sys
from math import sqrt
from numpy.random import default_rng
import cv2

# set random seed for random number generator
rng = default_rng(777)

# own coordinates for cube
cube_x_coord = 0 # max 8 = right of screen
cube_y_coord = 0 # max 6 = bottom of screen
cube_z_coord = 0
cube_scale = 1

# inital FP NN setup

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
    return (c_region_counters[largest_south_index] , r_region_counters[largest_south_index]) , np.array(loadedImage)


    

# API where can pass in 0-1 xy coords and have cube snap there
def moveCube(normalized_x_coord, normalized_y_coord, normalized_z_coord):
    # print("normalized_x_coord: ", normalized_x_coord)
    global cube_x_coord, cube_y_coord, cube_z_coord
    # 0 = left and top of screen
    # 1 = right and bottom of screen
    x_delta = (normalized_x_coord*8.0) - cube_x_coord
    y_delta = ((normalized_y_coord*6.0) - cube_y_coord)
    z_delta = (normalized_z_coord)- cube_z_coord
    cube_x_coord = normalized_x_coord*8.0
    cube_y_coord = normalized_y_coord*6.0
    glTranslatef(x_delta,y_delta, 0)

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
        # writeToFile("input_layer: " + str(self.output_neurons))

class Convolution:
    output_neurons = []
    filters = [] # set to rows, cols, channels, numFilters (3, 3, 1, 4) # kernelx, kernely, num_channels, num_filters
    bias = [] # 16 biases for 16 filters.
    delta = []
    
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

    def activateMatrix(self,matrix,bias):
        matrix = matrix + bias # add the bias to each value of a matrix
        return self.activation( matrix, self.activation_function)

    # convolution operation
    def convolve(self,input, filter):
        return scipy.signal.correlate2d(input, filter, mode='valid')

    def forward_propagation(self, previous_layer_output):
        conv_start_time = time.time()
        # filters = [] # set to rows, cols, channels, numFilters (3, 3, 1, 16) # kernelx, kernely, num_channels, num_filters
        # bias = [] # 16 biases for 16 filters.
        # input shape is (120, 115, 1)
        # output needs to be (None, 118, 113, 16) 
        output_size = ( (previous_layer_output.shape[0]-self.kernel_size[0]+1) , (previous_layer_output.shape[1]-self.kernel_size[1]+1), self.number_filters ) 
        self.output_neurons = np.zeros(output_size)
        # convolve each RGB channel with the relevant filter channel
        for num_filter in range(self.number_filters): # for each filter
            convolution_outputs = np.zeros( (output_size[0],output_size[1]) )
            for num_channel in range(previous_layer_output.shape[2]):
                # convolve each filter channel with each input image channel and sum all the channel outputs together
                convolution_outputs = convolution_outputs + self.convolve(previous_layer_output[:,:,num_channel], self.filters[:,:,num_channel,num_filter])
            # set each part of convolved output matrix after adding bias and activating
            self.output_neurons[:,:,num_filter] = self.activateMatrix(convolution_outputs, self.bias[num_filter])
        # print(f"Conv runtime: {(time.time()  - conv_start_time):.2f} seconds")
        # print("Shape at output of conv. layer 1: ", self.output_neurons.shape)
        # writeToFile("convolution_output: " + str(self.output_neurons))
        # writeToFile("convolution_output shape: " + str(self.output_neurons.shape))
        
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
        # output shape will be (58, 56, 16)
        # previous layer output is (118, 113, 16) 
        maxpool_start_time = time.time()
        # self.xy_max_pairs = np.zeros((int((((previous_layer_output.shape[0]-self.pool_size[0])/self.stride_size)+1)) , int((((previous_layer_output.shape[1]-self.pool_size[1])/self.stride_size)+1)) , previous_layer_output.shape[2], 2))
        self.output_neurons = np.zeros( ( int( (previous_layer_output.shape[0]-self.pool_size[0]) /self.stride_size ) +1 , int( (previous_layer_output.shape[1]-self.pool_size[1]) /self.stride_size ) +1 , previous_layer_output.shape[2]) ) # new dimensions after pooling
        max_pooled_values = []
        # inputX and inputY are counters over input layer
        inputX = 0
        inputY = 0
        # iterate over the entire input matrix
        while(inputX+self.pool_size[0] < previous_layer_output.shape[0]+1):
            while(inputY+self.pool_size[1] < previous_layer_output.shape[1]+1):
                for filter_num in range(previous_layer_output.shape[2]):
                    # filterX and filterY are the x and y positions going across the length of the imaginary pooling matrix
                    # iterate over the whole imaginary pooling matrix
                    maxValue = np.max(previous_layer_output[inputX:inputX+self.pool_size[0],inputY:inputY+self.pool_size[1],filter_num])
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
        flatten_time = time.time()
        self.output_neurons = np.array(previous_layer_output).flatten()
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
        #weights are (51968, 100)
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
        total_start_time = time.time()
        self.forward_propagation(new_input_data)
        # print(f"Total runtime: {(time.time()  - total_start_time):.2f} seconds")
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

### set up neural networks ###
model_fist_open = Network([
    Input("relu"),
    Convolution(16, (3,3), "relu"),
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(100, "relu"),
    Output(2, "softmax")])

model_directions = Network([
    Input("relu"),
    Convolution(16, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(500, "relu"),
    Output(3, "softmax")])

model_palm_side_back = Network([
    Input("relu"),
    Convolution(16, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(500, "relu"),
    Output(3, "softmax")])

# load weights from pre-trained TF NN
def load_parameters():
    global model_fist_open, model_directions, model_palm_side_back

    model_fist_open.layers[1].filters = np.load("Neural Network/tensorflow_gesture_detection_topdown_fist_open/conv_weights.npy")
    model_fist_open.layers[1].bias = np.load("Neural Network/tensorflow_gesture_detection_topdown_fist_open/conv_biases.npy")
    model_fist_open.layers[4].weights = np.load("Neural Network/tensorflow_gesture_detection_topdown_fist_open/hidden_weights.npy")
    model_fist_open.layers[5].weights = np.load("Neural Network/tensorflow_gesture_detection_topdown_fist_open/output_weights.npy")
    model_fist_open.layers[4].bias = np.load("Neural Network/tensorflow_gesture_detection_topdown_fist_open/hidden_biases.npy")
    model_fist_open.layers[5].bias = np.load("Neural Network/tensorflow_gesture_detection_topdown_fist_open/output_biases.npy")
    model_fist_open.networkInitialized = True

    model_directions.layers[1].filters = np.load("Neural Network/tensorflow_topdown_directions/conv_weights.npy")
    model_directions.layers[1].bias = np.load("Neural Network/tensorflow_topdown_directions/conv_biases.npy")
    model_directions.layers[4].weights = np.load("Neural Network/tensorflow_topdown_directions/hidden_weights.npy")
    model_directions.layers[5].weights = np.load("Neural Network/tensorflow_topdown_directions/output_weights.npy")
    model_directions.layers[4].bias = np.load("Neural Network/tensorflow_topdown_directions/hidden_biases.npy")
    model_directions.layers[5].bias = np.load("Neural Network/tensorflow_topdown_directions/output_biases.npy")
    model_directions.networkInitialized = True

    model_palm_side_back.layers[1].filters = np.load("Neural Network/tensorflow_topdown_palm_side_back/conv_weights.npy")
    model_palm_side_back.layers[1].bias = np.load("Neural Network/tensorflow_topdown_palm_side_back/conv_biases.npy")
    model_palm_side_back.layers[4].weights = np.load("Neural Network/tensorflow_topdown_palm_side_back/hidden_weights.npy")
    model_palm_side_back.layers[5].weights = np.load("Neural Network/tensorflow_topdown_palm_side_back/output_weights.npy")
    model_palm_side_back.layers[4].bias = np.load("Neural Network/tensorflow_topdown_palm_side_back/hidden_biases.npy")
    model_palm_side_back.layers[5].bias = np.load("Neural Network/tensorflow_topdown_palm_side_back/output_biases.npy")
    model_palm_side_back.networkInitialized = True

load_parameters()

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
            glVertex3fv(vertices[vertex])
    glEnd() # end of special OpenGL commands

def detect_current_gesture(frame):
    # Detect current gesture
        hand_bounding_box , loaded_image_new = loadImage(frame)
        loaded_image_new = np.array(loaded_image_new).reshape((30,30,1))  
        cv2.imshow('Extracted Hand', loaded_image_new)
        fist_prediction = model_fist_open.predict(loaded_image_new)
        direction_prediction = model_directions.predict(loaded_image_new)
        model_palm_side_back_prediction = model_palm_side_back.predict(loaded_image_new)
        prediction_text = "Prediction: "
        index_min = np.argmax(model_palm_side_back_prediction)
        if(index_min == 0):
            prediction_text += "Back "
        if(index_min == 1):
            prediction_text += "Side "
        if(index_min == 2):
            prediction_text += "Palm "
        index_min = np.argmax(direction_prediction)
        if(index_min == 0):
            prediction_text += "Up "
        if(index_min == 1):
            prediction_text += "Left "
        if(index_min == 2):
            prediction_text += "Right "
        index_min = np.argmax(fist_prediction)
        if(index_min == 0):
            prediction_text += "Open "
        if(index_min == 1):
            prediction_text += "Fist "
        return hand_bounding_box , prediction_text

# start a pygame window and modify it for displaying OpenGL content
def main():
    global cube_x_coord, cube_y_coord, cube_z_coord, cube_scale
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) # clear OpenGL canvas
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0) # set perspective of OpenGL window
    # 45deg fov, aspect ratio, near + far clipping planes (where cube visible)
    glTranslatef(-3.97,-2.7, -10.0) # move the cube to bottom left

    # main run loop
    while True:
        # display image

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        texture_background = glGenTextures(1) # create OpenGL texture
        
        vid = cv2.VideoCapture(1) # define a video capture object
        success, frame = vid.read() # Capture the video frame by frame
        if not success:
            print("Ignoring empty camera frame.")
            continue
        # get info of frame and convert to format needed for OpenGL texture
        background_image = cv2.flip(frame,0)
        background_image = Image.fromarray(background_image)
        image_width = background_image.size[0]
        image_height = background_image.size[1]
        background_image = background_image.tobytes('raw', 'BGRX', 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, background_image)

        # draw background2
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 1.0,  1.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,  1.0, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        
        glDisable(GL_DEPTH_TEST)

        prediction_time = time.time()
        hand_bounding_box , gesture_text = detect_current_gesture(frame) # hand gesture detection
        print(gesture_text,end="")
        print(f"Total runtime: {(time.time()  - prediction_time):.2f} seconds")
        if("Fist" in gesture_text):
            moveCube((hand_bounding_box[0]+40)/240,(320-(hand_bounding_box[1]+30))/320,0.6)
        if("Side" in gesture_text):
            
            glRotatef(5,-1,0,0) # Rotate perspective by angle, x, y and z.
        if("Palm" in gesture_text):
            glRotatef(5,0,1,0) # Rotate perspective by angle, x, y and z.
        # if("Right" in gesture_text):
        #     glRotatef(5,0,0,1) # Rotate perspective by angle, x, y and z.
        # if("Left" in gesture_text):
        #     glRotatef(5,0,0,-1) # Rotate perspective by angle, x, y and z.
              
        for event in pygame.event.get():
            if event.type == pygame.QUIT: # upon close of window
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                pass
                # if event.key == pygame.K_LEFT :
                #     glTranslatef(-1, 0, 0)
                #     cube_x_coord -= 1*cube_scale
                # if event.key == pygame.K_RIGHT :
                #     glTranslatef(1, 0, 0)
                #     cube_x_coord += 1*cube_scale
                # if event.key == pygame.K_UP :
                #     glTranslatef(0, 1, 0)
                #     cube_y_coord += 1*cube_scale
                # if event.key == pygame.K_DOWN :
                #     glTranslatef(0, -1, 0)
                #     cube_y_coord -= 1*cube_scale
                # if event.key == pygame.K_i :
                #     glScalef(1.5,1.5,1.5)
                #     glTranslatef(1/3, 1/3, 0)
                #     cube_scale *= 1.5
                # if event.key == pygame.K_o :
                #     glScalef(0.5,0.5,0.5)
                #     glTranslatef(-1, -1, 0)
                #     cube_scale *= 0.5
                # if event.key == pygame.K_w :
                #     glRotatef(45,1,0,0) # Rotate perspective by angle, x, y and z.
                # if event.key == pygame.K_s :
                #     glRotatef(45,-1,0,0) # Rotate perspective by angle, x, y and z.
                # if event.key == pygame.K_a :
                #     glRotatef(45,0,1,0) # Rotate perspective by angle, x, y and z.
                # if event.key == pygame.K_d :
                #     glRotatef(45,0,-1,0) # Rotate perspective by angle, x, y and z.
                
        Cube() # display the cube    

        # print("X: ", cube_x_coord, ", Y: ", cube_y_coord, ", Z: ", cube_z_coord, ", Scale: ", cube_scale)
        
        pygame.display.flip() # update the display
    
main()