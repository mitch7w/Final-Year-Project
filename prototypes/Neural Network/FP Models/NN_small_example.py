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
from numba import jit
conv_start_time=0
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

# set random seed for random number generator
rng = default_rng(777)

# set start time of program execution
start_time = time.time()

#open file to write out hyperparamaters and layer values to
details_file = open("NN_small_example.txt", "w+")
enableTextFileLogging = True
loadWeightsFromFile = True

# accepts arrays and strings and writes them in the network_details file
def writeToFile(input):
    if(enableTextFileLogging):
        details_file.write(str(input)+"\n\n")

imageReadIn = cv2.imread("/Users/mitch/Documents/University/Project/Project GitLab/prototypes/topdown_view_sizing.jpg",1) 
image = cv2.resize(imageReadIn, (10,7), interpolation = cv2.INTER_AREA)
loadedImage = np.array(image)/255

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
        output_size = ( (previous_layer_output.shape[0]-self.kernel_size[0]+1) , (previous_layer_output.shape[1]-self.kernel_size[1]+1), self.number_filters ) # define output size based on kernel and stride size
        self.output_neurons = np.zeros(output_size)

        # # optimised implementation
        im_col = np.lib.stride_tricks.as_strided(
                previous_layer_output, (previous_layer_output.shape[0]-self.filters.shape[0]+1, previous_layer_output.shape[1]-self.filters.shape[1]+1, self.filters.shape[0], self.filters.shape[1], previous_layer_output.shape[2]), 
                previous_layer_output.strides[:2] + previous_layer_output.strides[0:], writeable=False)
        image_multiplication = np.tensordot(im_col, self.filters, axes=3)
        self.output_neurons = self.activateMatrix(image_multiplication, self.bias)
        print(f"Conv runtime: {(time.time()  - conv_start_time):.2f} seconds")
        # print("Shape at output of conv. layer 1: ", self.output_neurons.shape)
        writeToFile("convolution_output: " + str(self.output_neurons))
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
        self.output_neurons = np.zeros( ( int( (previous_layer_output.shape[0]-self.pool_size[0]) /self.stride_size ) +1 , int( (previous_layer_output.shape[1]-self.pool_size[1]) /self.stride_size ) +1 , previous_layer_output.shape[2]) ) # new dimensions after pooling
        
        # optimised routine
        for prev_layer_channel in range(previous_layer_output.shape[2]):
            strided_pool_groups = np.lib.stride_tricks.sliding_window_view(previous_layer_output[:,:,prev_layer_channel], self.pool_size)[::self.stride_size,::self.stride_size]
            strided_pool_groups_flattened = strided_pool_groups.reshape((strided_pool_groups.shape[0]*strided_pool_groups.shape[1],strided_pool_groups.shape[2]*strided_pool_groups.shape[3]))
            max_pooled_values = np.max(strided_pool_groups_flattened, axis=1)
            self.output_neurons[:,:,prev_layer_channel] = np.array(max_pooled_values).reshape(( int( (previous_layer_output.shape[0]-self.pool_size[0]) /self.stride_size ) +1 , int( (previous_layer_output.shape[1]-self.pool_size[1]) /self.stride_size ) +1 ))
        # print("Shape at output of Maxpooling layer: ", self.output_neurons.shape)

        print(f"Maxpool runtime: {(time.time()  - maxpool_start_time):.2f} seconds")
        writeToFile("Maxpooling layer: " + str(self.output_neurons))
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
        writeToFile("Flatten layer: " + str(self.output_neurons))

        print(f"Flatten runtime: {(time.time()  - flatten_time):.2f} seconds")

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
        print(f"Hidden runtime: {(time.time()  - hidden_time):.2f} seconds")
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
        print(f"Output runtime: {(time.time()  - output_time):.2f} seconds")
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
        total_start_time = time.time()
        self.forward_propagation(new_input_data)
        print(f"Total runtime: {(time.time()  - total_start_time):.2f} seconds")
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

### set up neural network ###
model = Network([
    Input("relu"),
    Convolution(4, (3,3), "relu"), # num_filters, (filter_x x filter_y x num_channels)
    Maxpooling((3,3),2),
    Flatten(),
    Hidden(5, "relu"),
    Output(2, "softmax")])

# load weights from pre-trained NN if so desired
if(loadWeightsFromFile):
    model.layers[1].filters = np.load("tensorflow_small_example/conv_weights.npy") # kernelx, kernely, num_channels, num_filters
    model.layers[1].bias = np.load("tensorflow_small_example/conv_biases.npy")
    model.layers[4].weights = np.load("tensorflow_small_example/hidden_weights.npy")
    model.layers[5].weights = np.load("tensorflow_small_example/output_weights.npy")
    model.layers[4].bias = np.load("tensorflow_small_example/hidden_biases.npy")
    model.layers[5].bias = np.load("tensorflow_small_example/output_biases.npy")
    model.networkInitialized = True

new_predict = model.predict(loadedImage)
exit()


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
    loaded_image_new = np.array(loadImage(image)).reshape((120,115,1))
    # print("loaded_image_new.shape: ", loaded_image_new.shape)
    cv2.imshow('Extracted Hand', loaded_image_new)
    new_prediction = model.predict(loaded_image_new)
    index_min = np.argmax(new_prediction)
    if(index_min == 0):
        print("Prediction: Open\n\n")
    else:
        print("Prediction: Fist\n\n")
    if cv2.waitKey(5) == ord(" "):
        break

cap.release()


details_file.close() # close network details textfile

# end program runtime and report duration
end_time = time.time() 
print(f"Program runtime: {(end_time - start_time):.2f} seconds")

# original output
# input_layer: [[[0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]]

#  [[0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.25490196 0.25490196 0.25490196]
#   [0.03921569 0.03921569 0.03921569]
#   [0.20392157 0.20392157 0.20392157]
#   [0.18039216 0.18039216 0.18039216]
#   [0.04313725 0.04313725 0.04313725]
#   [0.01176471 0.01176471 0.01176471]
#   [0.05882353 0.05882353 0.05882353]
#   [0.         0.         0.        ]]

#  [[0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.22745098 0.22745098 0.22745098]
#   [0.23529412 0.23529412 0.23529412]
#   [0.84313725 0.84313725 0.84313725]
#   [0.74117647 0.74117647 0.74117647]
#   [0.33333333 0.33333333 0.33333333]
#   [0.17647059 0.17647059 0.17647059]
#   [0.04705882 0.04705882 0.04705882]
#   [0.         0.         0.        ]]

#  [[0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.20784314 0.20784314 0.20784314]
#   [0.62745098 0.62745098 0.62745098]
#   [0.81568627 0.81568627 0.81568627]
#   [0.24313725 0.24313725 0.24313725]
#   [0.07058824 0.07058824 0.07058824]
#   [0.08627451 0.08627451 0.08627451]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]]

#  [[0.         0.         0.        ]
#   [0.40784314 0.40784314 0.40784314]
#   [0.97647059 0.97647059 0.97647059]
#   [0.32941176 0.32941176 0.32941176]
#   [0.         0.         0.        ]
#   [0.05098039 0.05098039 0.05098039]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]]

#  [[0.23921569 0.23921569 0.23921569]
#   [0.99215686 0.99215686 0.99215686]
#   [0.57647059 0.57647059 0.57647059]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]]

#  [[0.39215686 0.39215686 0.39215686]
#   [0.4        0.4        0.4       ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.00392157 0.00392157 0.00392157]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]
#   [0.         0.         0.        ]]]

# convolution_output: [[[-0.00000000e+00  3.73645947e-01  1.65651843e-01 -0.00000000e+00]
#   [ 9.22254850e-02  4.90886857e-01  3.33555147e-01 -0.00000000e+00]
#   [-0.00000000e+00  9.82670409e-01  7.17658431e-01 -0.00000000e+00]
#   [ 6.92310871e-04  1.22892539e+00  8.70574598e-01 -0.00000000e+00]
#   [-0.00000000e+00  6.71296157e-01  1.04529042e+00 -0.00000000e+00]
#   [-0.00000000e+00  2.73298807e-01  7.30296673e-01 -0.00000000e+00]
#   [-0.00000000e+00  1.49228357e-01  3.35067113e-01 -0.00000000e+00]
#   [ 2.09905048e-02  5.93601815e-02  1.93152946e-01 -0.00000000e+00]]

#  [[-0.00000000e+00  4.36423815e-01  1.47622915e-01 -0.00000000e+00]
#   [ 5.23194395e-02  8.86048256e-01  4.94569430e-01 -0.00000000e+00]
#   [-0.00000000e+00  1.57208262e+00  8.71783453e-01 -0.00000000e+00]
#   [-0.00000000e+00  1.43349020e+00  1.23763292e+00 -0.00000000e+00]
#   [-0.00000000e+00  6.32257969e-01  1.12083828e+00 -0.00000000e+00]
#   [-0.00000000e+00  3.56317610e-01  5.07775585e-01 -0.00000000e+00]
#   [-0.00000000e+00  2.08627609e-01  2.37192849e-01 -0.00000000e+00]
#   [-0.00000000e+00  4.26784537e-02  1.37468418e-01 -0.00000000e+00]]

#  [[-0.00000000e+00  1.25300336e+00  6.12545777e-01 -0.00000000e+00]
#   [-0.00000000e+00  1.22835358e+00  8.26389372e-01 -0.00000000e+00]
#   [-0.00000000e+00  1.04062065e+00  1.06189047e+00 -0.00000000e+00]
#   [-0.00000000e+00  6.74342593e-01  6.55607777e-01 -0.00000000e+00]
#   [-0.00000000e+00 -0.00000000e+00  6.56726563e-02 -0.00000000e+00]
#   [-0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00]
#   [ 6.30765197e-02 -0.00000000e+00  7.54444948e-03 -0.00000000e+00]
#   [-0.00000000e+00 -0.00000000e+00  1.19052218e-02 -0.00000000e+00]]

#  [[-0.00000000e+00  1.80691178e+00  9.83498578e-01 -0.00000000e+00]
#   [-0.00000000e+00  1.09226256e+00  1.35229138e+00 -0.00000000e+00]
#   [-0.00000000e+00  3.06114173e-01  6.11082910e-01 -0.00000000e+00]
#   [-0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00]
#   [ 1.83788657e-01 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00]
#   [ 2.97007487e-02 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00]
#   [ 5.30034621e-02 -0.00000000e+00  1.64534683e-02 -0.00000000e+00]
#   [ 4.08053769e-02 -0.00000000e+00  2.02094986e-02  7.82755364e-03]]

#  [[-0.00000000e+00  1.33216720e+00  8.98950304e-01 -0.00000000e+00]
#   [-0.00000000e+00  1.27936931e-01  5.64451066e-01 -0.00000000e+00]
#   [-0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00]
#   [ 7.15295948e-02 -0.00000000e+00 -0.00000000e+00  4.81462860e-02]
#   [ 3.82709302e-02  2.17915990e-02  3.62907958e-02 -0.00000000e+00]
#   [ 3.65177485e-02  7.79047952e-03  2.74478359e-02 -0.00000000e+00]
#   [ 3.03245075e-02  3.10688503e-02  3.79032120e-02 -0.00000000e+00]
#   [ 3.03245075e-02  3.10688503e-02  3.79032120e-02 -0.00000000e+00]]]

# Maxpooling layer: [[[ 9.22254850e-02  1.57208262e+00  1.06189047e+00 -0.00000000e+00]
#   [ 6.92310871e-04  1.57208262e+00  1.23763292e+00 -0.00000000e+00]
#   [ 6.30765197e-02  6.71296157e-01  1.12083828e+00 -0.00000000e+00]]

#  [[-0.00000000e+00  1.80691178e+00  1.35229138e+00 -0.00000000e+00]
#   [ 1.83788657e-01  1.04062065e+00  1.06189047e+00  4.81462860e-02]
#   [ 1.83788657e-01  3.10688503e-02  6.56726563e-02 -0.00000000e+00]]]

# Flatten layer: [ 9.22254850e-02  1.57208262e+00  1.06189047e+00 -0.00000000e+00
#   6.92310871e-04  1.57208262e+00  1.23763292e+00 -0.00000000e+00
#   6.30765197e-02  6.71296157e-01  1.12083828e+00 -0.00000000e+00
#  -0.00000000e+00  1.80691178e+00  1.35229138e+00 -0.00000000e+00
#   1.83788657e-01  1.04062065e+00  1.06189047e+00  4.81462860e-02
#   1.83788657e-01  3.10688503e-02  6.56726563e-02 -0.00000000e+00]

# Hidden layer: [0.3642658101393056, 1.3130092885435072, 2.4026201066901467, -0.0, 1.7841137907324316]

# Output layer: [0.99209803 0.00790197]

