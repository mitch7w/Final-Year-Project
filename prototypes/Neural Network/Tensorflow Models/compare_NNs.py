import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers, callbacks
import scipy.signal
import cv2
from numpy.random import default_rng
# set random seed for random number generator
rng = default_rng(777)
import time
import random
import sys
import freenect
from PIL import ImageEnhance, Image

# FP NN

class Input:
    output_neurons = []
  
    def __init__(self, activation_function):
        self.activation_function = activation_function

    def get_size(self):
        return len(self.output_neurons)

    def get_output(self):
        return self.output_neurons

    def forward_propagation(self,new_neurons):
        self.output_neurons = new_neurons.copy()
        # writeToFile("input_layer: " + str(self.output_neurons))

class Convolution:
    output_neurons = []
    filters = [] # filter values in shape of (rows, cols, channels, numFilters). E.g. 3x3 greyscale with 4 filters = (3, 3, 1, 4)
    bias = [] # 1 bias for each filter
    delta = [] # delta values to backpropagate error
    
    # initialise values
    def __init__(self, number_filters, kernel_size, activation_function,num_channels):
        self.number_filters = number_filters
        self.kernel_size = kernel_size
        self.activation_function = activation_function
        # initialize filters with random values from -1 to 1
        self.filters = []
        for x in range((kernel_size[0]*kernel_size[1]*number_filters*num_channels)):
            self.filters.append(0.01)
        self.bias = []
        for x in range(self.number_filters):
            self.bias.append(0.01)
        for f in range(len(self.filters)):
            self.filters[f] = (self.filters[f] *2) -1  # generate new filter value -1 to 1
        self.filters=np.array(self.filters).reshape(kernel_size[0],kernel_size[1],num_channels,number_filters)
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

    # activate each value in a matrix with chosen activation function
    def activate_matrix(self,matrix,bias):
        matrix = matrix + bias # add the bias to each value of a matrix
        return self.activation( matrix, self.activation_function)

    # convolution operation for an input image and filter
    def convolve(self,input, filter):
        return scipy.signal.correlate2d(input, filter, mode='valid')

    # slide filters over input image and get convolved output
    def forward_propagation(self, previous_layer_output):
        # print("monkeys2: ", previous_layer_output)
        conv_start_time = time.time()
        # filter values in shape of (rows, cols, channels, numFilters). E.g. 3x3 greyscale with 4 filters = (3, 3, 1, 4)
        # bias = 1 bias for each filter
        output_size = ( (previous_layer_output.shape[0]-self.kernel_size[0]+1) , (previous_layer_output.shape[1]-self.kernel_size[1]+1), self.number_filters ) # define output size based on kernel and stride size
        self.output_neurons = np.zeros(output_size)
        # convolve each RGB channel with the relevant filter channel
        for num_filter in range(self.number_filters): # for each filter
            convolution_outputs = np.zeros( (output_size[0],output_size[1]) )
            for num_channel in range(previous_layer_output.shape[2]):
                # convolve each filter channel with each input image channel and sum all the channel outputs together
                convolution_outputs = convolution_outputs + self.convolve(previous_layer_output[:,:,num_channel], self.filters[:,:,num_channel,num_filter])
            # set each part of convolved output matrix after adding bias and activating
            self.output_neurons[:,:,num_filter] = self.activate_matrix(convolution_outputs, self.bias[num_filter])
        # print(f"Conv runtime: {(time.time()  - conv_start_time):.2f} seconds")
        # print("Shape at output of conv. layer 1: ", self.output_neurons.shape)
        # writeToFile("convolution_output: " + str(self.output_neurons))
        # writeToFile("convolution_output shape: " + str(self.output_neurons.shape))
        
    # propagate error backwards through convolution layer
    def backpropagation(self, next_layer_delta, xy_max_pairs, input_data,learning_rate):
        convolution_gradient = np.zeros(self.output_neurons.shape[0]*self.output_neurons.shape[1]*self.number_filters*len(input_data[0][0])).reshape((self.output_neurons.shape[0],self.output_neurons.shape[1],self.number_filters,len(input_data[0][0])))
        # for every max-pooled value set its gradient based on next layer delta
        next_layer_delta = np.array(next_layer_delta).flatten()
        next_layer_delta_counter = 0
        for num_filter in range(self.number_filters):
            for x in range(len(xy_max_pairs)):
                for y in range(len(xy_max_pairs[x])):
                    convolution_gradient[int(xy_max_pairs[x][y][num_filter][0])][int(xy_max_pairs[x][y][num_filter][1])][num_filter] = next_layer_delta[next_layer_delta_counter]
                    next_layer_delta_counter +=1
        # find filter gradients based on delta values and input and update filter values for each R,G,B filter
        filter_gradients = np.empty(self.filters.shape)
        for num_filter in range(self.number_filters):
            channel_gradients = np.empty((self.filters.shape[1],self.filters.shape[2],self.filters.shape[3]))
            for channel_num in range(len(input_data[0][0])):
                channel_gradients[:,:,channel_num] = self.convolve(input_data[:,:,channel_num],convolution_gradient[:,:,num_filter,channel_num])

            filter_gradients[num_filter] = channel_gradients
        # update each filter value
        for num_filter in range(len(self.filters)):
            for channel_num in range(len(input_data[0][0])):
                for x in range(len(self.filters[num_filter])):
                    for y in range(len(self.filters[num_filter][x])):
                        self.filters[num_filter][x][y][channel_num] -= learning_rate * filter_gradients[num_filter][x][y][channel_num]
        # update input matrix X with full convolvution of 180deg rotated filter and gradient
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
        # some operations commented out to save processing time when not doing backprop.
        maxpool_start_time = time.time()
        # set size of variables based on stride, pool and input size
        # self.xy_max_pairs = np.zeros((int((((previous_layer_output.shape[0]-self.pool_size[0])/self.stride_size)+1)) , int((((previous_layer_output.shape[1]-self.pool_size[1])/self.stride_size)+1)) , previous_layer_output.shape[2], 2))
        self.output_neurons = np.zeros( ( int( (previous_layer_output.shape[0]-self.pool_size[0]) /self.stride_size ) +1 , int( (previous_layer_output.shape[1]-self.pool_size[1]) /self.stride_size ) +1 , previous_layer_output.shape[2]) ) # new dimensions after pooling
        max_pooled_values = []
        # inputX and inputY are counters over input layer
        inputX = 0
        inputY = 0
        # iterate over the entire input matrix in pools of pool_size[0] x pool_size[1]
        while(inputX+self.pool_size[0] < previous_layer_output.shape[0]+1):
            while(inputY+self.pool_size[1] < previous_layer_output.shape[1]+1):
                for filter_num in range(previous_layer_output.shape[2]):
                    # filterX and filterY are the x and y positions going across the length of the pooling matrix
                    # iterate over the whole pooling matrix
                    maxValue = np.max(previous_layer_output[inputX:inputX+self.pool_size[0],inputY:inputY+self.pool_size[1],filter_num]) # find max value in pool
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
        self.delta = next_layer_delta # just pass previous error backwards since this layer has no learned parameters

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
        self.output_neurons = np.array(previous_layer_output).flatten() # turn multi-dimensional array into 1D array
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
        # set weights and biases to random values
        self.weights = []
        for x in range(previous_layer_output_size*self.number_neurons):
            self.weights.append( random.random())
        self.bias = []
        for x in range(self.number_neurons):
            self.bias.append( random.random())
        # standard_deviation = np.sqrt(2/previous_layer_output_size) # find std based on input size
        # modify weights and biases with the std
        for w in range(len(self.weights)):
            self.weights[w] = (self.weights[w] * 2) - 1
        self.weights=np.array(self.weights).reshape(self.number_neurons,previous_layer_output_size)
        for b in range(len(self.bias)):
            self.bias[b] = (self.bias[b] * 2) -1

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
        #weights are (input_connections, num_hidden_neurons)
        self.output_neurons=[]
        for neuron_number in range(self.number_neurons):
            # update hidden layer with dot product of inputs * weights all run through the activation function
            activationInput = np.dot(previous_layer_output , self.weights[neuron_number,:])+ self.bias[neuron_number]
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
        # update weights
        for i in range(len(self.weights)): # self.weights[i] = each hidden neuron
            for j in range(len(self.weights[i])): # self.weights[i][j] = hidden neuron's connection to each previous layer jth neuron
                self.weights[i][j] -= learning_rate * self.hidden_delta[i] * previous_layer_output[j]
            # update biases
            self.bias[i] -= learning_rate * self.hidden_delta[i]
        # writeToFile("hidden_layer_weights: " + str(self.weights))

    def get_weights(self):
        return self.weights

    def get_delta(self):
        return self.hidden_delta
    
    def get_biases(self):
        return self.bias

class Output:
    # Only difference from hidden is how backpropagation is conducted - now with desired training output
    output_neurons = []
    output_delta = []

    def __init__(self, number_neurons, activation_function):
        self.number_neurons = number_neurons
        self.activation_function = activation_function
    
    def get_size(self):
        return self.number_neurons

    # initialize weights and biases with the He method
    def initialize_He_weights(self,previous_layer_output_size):
        # set weights and biases to random values
        self.weights = []
        for x in range(previous_layer_output_size*self.number_neurons):
            self.weights.append( random.random())
        self.bias = []
        for x in range(self.number_neurons):
            self.bias.append( random.random())
        # standard_deviation = np.sqrt(2/previous_layer_output_size) # find std based on input size
        # modify weights and biases with the std
        for w in range(len(self.weights)):
            self.weights[w] = (self.weights[w] * 2) - 1
        self.weights=np.array(self.weights).reshape(self.number_neurons,previous_layer_output_size)
        for b in range(len(self.bias)):
            self.bias[b] = (self.bias[b] * 2) -1

    def get_output(self):
        return self.output_neurons

    def activation(self, output_array, activation_function):
        if(activation_function=="softmax"):
            exponents_output_array = np.exp(output_array)
            print("output_array: ", output_array)
            print("exponents_output_array: ", exponents_output_array)
            return exponents_output_array / np.sum(exponents_output_array)

    def activation_derivative(self, x, activation_function):
        print("x: ", x)
        return x * (x > 0)
    def forward_propagation(self, previous_layer_output):
        # print("monkey1: ", previous_layer_output)
        output_time = time.time()
        self.output_neurons=[]
        for neuron_number in range(self.number_neurons):
            # update hidden layer with dot product of inputs * weights all run through the activation function
            activationInput = np.dot(previous_layer_output , self.weights[neuron_number,:])+ self.bias[neuron_number]
            self.output_neurons.append(activationInput)
        self.output_neurons = self.activation(self.output_neurons, self.activation_function) # activate output
        print("hereoutputnerons: ", self.output_neurons)
        # print(f"Output runtime: {(time.time()  - output_time):.2f} seconds")
        # writeToFile("Output layer: " + str(self.output_neurons))

    def backpropagation(self,desired_output,learning_rate,previous_layer_output):
        # calculate errors for output layer
        output_errors = np.array(self.output_neurons) - np.array(desired_output)
        # calculate delta of output layer - how much they must change
        self.output_delta = []
        for i in range(len(output_errors)):
            print("self.output_neurons[i]: ", self.output_neurons)
            print("self.activation_derivative(self.output_neurons[i],self.activation_function): ", self.activation_derivative(self.output_neurons[i],self.activation_function))
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
    networkInitialized = False # once forward prop. is run once and all paramaters initialised, this is True

    def __init__(self, layers):
        self.layers = layers

    # perform forward propogation of whole network based on new input data
    def forward_propagation(self, new_input_data):
        for i in range(len(self.layers)):
            if(i==0):
                self.layers[i].forward_propagation(new_input_data) # pass in the new input data
                print("Layer output: ", self.layers[i].get_output())
            else:
                # Initialize weights with He method if first time through network
                if(self.networkInitialized == False):
                    if(isinstance(self.layers[i], Hidden) or isinstance(self.layers[i], Output)):
                        self.layers[i].initialize_He_weights(self.layers[i-1].get_size())
                # propagate forwards with previous layer's output as input to next layer
                self.layers[i].forward_propagation(self.layers[i-1].get_output())
                print("Layer output: ", self.layers[i].get_output())
        self.networkInitialized = True
        
    # run a new input image through forward prop. and get a prediction
    def predict(self, new_input_data):
        total_start_time = time.time()
        self.forward_propagation(new_input_data)
        # print(f"Total runtime: {(time.time()  - total_start_time):.2f} seconds")
        return self.layers[len(self.layers)-1].get_output()

    # backprop. the error of a desired output through the whole network and update all parameters
    def backpropagation(self, input_data, desired_output,learning_rate):
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
    
    # take in a dataset and train the network parameters based on the data
    def train(self, training_data_input, training_data_output, learning_rate, num_training_epochs):
        # writeToFile("learning_rate: " + str(learning_rate))
        # writeToFile("num_training_epochs: " + str(num_training_epochs))
        # the networks sees all the data num_training_epochs times
        for epoch in range(num_training_epochs):
            epoch_start_time = time.time()
            total_error_at_output = 0
            # for each set of training data calculate the total error
            length_training_data = len(training_data_input)
            for training_data_num in range(length_training_data):
                # print("monkeys3: ", training_data_input[training_data_num])
                self.forward_propagation(training_data_input[training_data_num])
                total_error_at_output += sum( (np.array(training_data_output[training_data_num]) -np.array(self.layers[len(self.layers)-1].get_output()))**2)
                self.backpropagation(training_data_input[training_data_num], training_data_output[training_data_num],learning_rate)        
                currentProgressString = str(training_data_num) + "/", str(length_training_data)
                print(training_data_num, "/", length_training_data, end='\r')
            epoch_end_time = time.time()
            # print current training progress
            print("Epoch: ", epoch, end=", ")
            print("Learning rate: ", learning_rate, end=", ")
            print("Error: ", total_error_at_output, end=", ")
            print(f"Runtime: {(epoch_end_time - epoch_start_time):.2f} seconds")
            self.epochErrors.append(total_error_at_output)
            # save weights for later use
            if(total_error_at_output < 1):
                np.save('Trained Nets/NN_gesture_layers_hidden_weights', model.layers[2].get_weights())
                np.save('Trained Nets/NN_gesture_layers_output_weights', model.layers[3].get_weights())
                np.save('Trained Nets/NN_gesture_layers_hidden_bias', model.layers[2].get_biases())
                np.save('Trained Nets/NN_gesture_layers_output_bias', model.layers[3].get_biases())
        # plot loss curve
        plt.plot(range(len(self.epochErrors)), self.epochErrors)
        plt.xlabel("Num Epochs")
        plt.ylabel("Training Square Loss")
        plt.show()

        # Overfitting dataset checks
        for i in range(len(training_data_input)):
            prediction = self.predict(training_data_input[i])
            print("Expected: ", training_data_output[i], ", Got: ", prediction)

# end of neural network layer classes
####################################################################################

### set up neural networks for AR application ###
model_fist_open = Network([
    Input("relu"),
    Convolution(16, (9,9), "relu", 3),
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(100, "relu"),
    Output(2, "softmax")])

####

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
    img2 = converter.enhance(0.5)
    # For reversing the operation:
    image = np.asarray(img2)
    # cv2.imshow("Enhancer", image)
    image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
    # cv2.imshow("Enhancer", image)
    # print(image_ycrcb[0][0])
    # skin1 = (0, 100, 136)
    # skin2 = (255, 144, 156)
    skin1 = (91, 94, 134)
    skin2 = (255, 131, 160)
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
    for c_counter in range(0,80,5):
        for r_counter in range(0,60,5):
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

    # widen extracted window to get the whole hand in the frame
    if(edge_max_index==0): # hand coming in from top
        if(crop_coords[0]-5 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 5

        if(crop_coords[1]+25 >79): # how much extra right to do
            crop_coords[1] = 79
        else:
            crop_coords[1] += 25

        if(crop_coords[2] - 20 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 20

        if(crop_coords[3]+20 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 20
        
    if(edge_max_index==1): # hand coming in from left
        if(crop_coords[0]-15 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 15

        if(crop_coords[1]+18 >79): # how much extra right to do
            crop_coords[1] = 79
        else:
            crop_coords[1] += 18

        if(crop_coords[2] - 10 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 10

        if(crop_coords[3]+20 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 20
        
    if(edge_max_index==2): # hand coming in from bottom
        if(crop_coords[0]-3 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 3
        if(crop_coords[1]+43 >59): # how much extra right to do
            crop_coords[1] = 59
        else:
            crop_coords[1] += 43
        if(crop_coords[2] - 10 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 10
        if(crop_coords[3]+30 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 30
        
    if(edge_max_index==3): # hand coming in from right
        if(crop_coords[0]-7 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 7
        if(crop_coords[1]+37 >79): # how much extra right to do
            crop_coords[1] = 79
        else:
            crop_coords[1] += 37
        if(crop_coords[2] - 10 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 10
        if(crop_coords[3]+30 >59): # how much extra down to do
            crop_coords[3] = 59
        else:
            crop_coords[3] += 30

    # extracted_hand = image[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]] 
    extracted_hand = result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]] # return Ycbcr-masked image
    # print("extracted_hand.shape: ", extracted_hand.shape)
    if(extracted_hand.shape != (40, 44, 3)):
        extracted_hand = cv2.resize(result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]], (44,40), interpolation = cv2.INTER_AREA)
        # extracted_hand = cv2.resize(image[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]], (44,40), interpolation = cv2.INTER_AREA)
    # extracted_hand = image_ycrcb # overwrite and just return Ycbcr color image
    # extracted_hand_greyscale = cv2.cvtColor(extracted_hand,cv2.COLOR_RGB2GRAY)
    # cv2.imshow("extracted_hand_greyscale", extracted_hand_greyscale)
    # cv2.waitKey(1)
    # make sure all images are same size even if cropped on the side
    loadedImage = np.array(extracted_hand)/255 # divide by 255 in order to normalize input
    if(len(loadedImage.shape) == 2): # if it's not 3 channels long put it in a 3D shape
        loadedImage = loadedImage.reshape((loadedImage.shape[0],loadedImage.shape[1],1))
    return np.array(loadedImage)

# load images
training_data_input = []
for i in range(200,202):
    imageName = "/Users/mitch/Documents/University/Project/fist_open_speed/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImageSmall(imageName))
for i in range(1505,1507):
    imageName = "/Users/mitch/Documents/University/Project/fist_open_speed/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImageSmall(imageName))
print("training_data_input shape: ", np.array(training_data_input).shape)

# load output classes
training_data_output = []
for j in range(1):
    for i in range(2):
            training_data_output.append([1,0]) # Open
    for i in range(2):
        training_data_output.append([0,1]) # Fist
    
print("training_data_output shape: ", np.array(training_data_output).shape)

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data

x_train = np.array(training_data_input[:-1])
y_train = np.array(training_data_output[:-1])
x_test = np.array(training_data_input[-1:])
y_test = np.array(training_data_output[-1:])
print("x_train.shape: ", x_train.shape)

### train network ###
model_fist_open.train(training_data_input , training_data_output, learning_rate=0.01, num_training_epochs=10)


# num_output_classes = 2
# input_shape = training_data_input[0].shape

# # Build the model
# model = keras.Sequential(
#     [
#         keras.Input(shape=input_shape),
#         layers.Conv2D(16, kernel_size=(9,9), activation="relu"),
#         layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
#         layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
#         layers.Flatten(),
#         layers.Dense(100, activation="relu"),
#         layers.Dense(num_output_classes, activation="softmax"), 
#     ]
# )

# model.summary()

# #Train the model
# batch_size = 1
# epochs = 6
# # Setup model
# optimizer = keras.optimizers.Adam(lr=0.001)
# model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])

# # # Early stopping
# # callback = callbacks.EarlyStopping(monitor='accuracy', patience=1)
# # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[callback],verbose=1)

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1,verbose=1)

# # Evaluate the model
# score = model.evaluate(x_test, y_test, verbose=0)
# print("Test loss:", score[0])
# print("Test accuracy:", score[1])