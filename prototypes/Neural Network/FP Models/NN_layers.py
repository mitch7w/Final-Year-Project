import numpy as np
import matplotlib.pyplot as plt
import time
from PIL import Image
import scipy.signal
import sys
from math import sqrt, isclose
from numpy.random import default_rng

# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

# set random seed for random number generator
rng = default_rng(777)

# set start time of program execution
start_time = time.time()

#open file to write out hyperparamaters and layer values to
details_file = open("network_details_nn_layers.txt", "w+")
enableTextFileLogging = True

# accepts arrays and strings and writes them in the network_details file
def writeToFile(input):
    if(enableTextFileLogging):
        details_file.write(str(input)+"\n\n")

# Load image, remove alpha channel and normalize input - change this method as needed for specific input data
def loadImage(imageName):
    # load image and remove alpha channel
    with Image.open(imageName) as im:
        loadedImage = np.array(im)/255 # divide by 255 in order to normalize input
        new3DImage = np.empty((28,28,3))
        for x in range(len(new3DImage)):
            for y in range(len(new3DImage[x])):
                new3DImage[x][y] = [loadedImage[x][y],loadedImage[x][y],loadedImage[x][y]]
        return new3DImage

# Different layers available for use in neural network
class Input:
    output_neurons = []
  
    def __init__(self, number_neurons, activation_function):
        self.number_neurons = number_neurons
        self.activation_function = activation_function

    def get_size(self):
        return self.number_neurons

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
        # initialize filters with random values from -1 to 1
        for i in range(number_filters):
            filter_init_values = rng.random(kernel_size[0]*kernel_size[1])
            for f in range(len(filter_init_values)):
                filter_init_values[f] = (filter_init_values[f] *2) -1
            self.filters.append(filter_init_values.reshape(kernel_size[0],kernel_size[1]))

    def get_size(self):
            return self.number_neurons

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
        # convolve each RGB channel with the relevant filter
        convolution_outputs = []
        for i in range(self.number_filters):
            convolution_outputs.append(self.convolve(previous_layer_output[:,:,i], self.filters[i]))
        # sum the results of each R,G,B convolution for the output convolution
        self.output_neurons = np.empty((convolution_outputs[0].shape[0], convolution_outputs[0].shape[1]))
        for x in range(convolution_outputs[0].shape[0]):
            for y in range(convolution_outputs[0].shape[1]):
                newTotal = 0
                for i in range(self.number_filters):
                    newTotal += convolution_outputs[i][x][y] # sum each rgb channel
                self.output_neurons[x][y] = newTotal
                # self.output_neurons[x][y] = self.output_neurons[x][y] / 765 # maybe neurons have to be 0-1
                # maybe add bias to convolutional adding
        writeToFile("convolution_output: " + str(self.output_neurons))
        
    def backpropagation(self, next_layer_delta, xy_max_pairs, input_data,learning_rate):
        convolution_gradient = np.zeros(self.output_neurons.shape[0]*self.output_neurons.shape[1]).reshape((self.output_neurons.shape[0],self.output_neurons.shape[1]))
        # for every max-pooled value set its gradient
        for i in range(len(xy_max_pairs)):
            convolution_gradient[xy_max_pairs[i][0],xy_max_pairs[i][1]] = next_layer_delta[i]
        # find filter gradients and update filter values for each R,G,B filter
        filter_gradients = []
        for i in range(self.number_filters):
            filter_gradients.append(self.convolve(input_data[:,:,i],convolution_gradient))
        # update filter values with F -= learningRate *dL/dF
        for i in range(len(self.filters)):
            for x in range(len(self.filters[i])):
                for y in range(len(self.filters[i][x])):
                    self.filters[i][x][y] -= learning_rate * filter_gradients[i][x][y]
        # update input matrix X with dL/dX = Full convolvution of 180deg rotated Filter F and loss gradient dL/dO
        rotatedFilters = []
        for i in range(len(self.filters)):
            newFilter = []
            for x in range(len(self.filters[i])):
                for y in range(len(self.filters[i][x])):
                    newFilter.append(self.filters[i][x][y])
            rotatedFilters.append(newFilter)
        # rotate filter about vertical then horizontal axis
        for i in range(len(rotatedFilters)):
            rotatedFilters[i].reverse()
            rotatedFilters[i] = np.array(rotatedFilters[i]).reshape((self.filters[i].shape[0],self.filters[i].shape[1]))
        inputDataDeltas = []
        for i in range(len(rotatedFilters)):
            inputDataDeltas.append(self.fullConvolution(rotatedFilters[i],convolution_gradient))
        self.delta = [0] * len(inputDataDeltas[0])
        for i in range(len(inputDataDeltas)):
            for j in range(len(inputDataDeltas[i])):
                self.delta[j] += inputDataDeltas[i][j]
        writeToFile("Filters: " + str(self.filters))

    def get_output(self):
        return self.output_neurons

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
        # dimensions of previous layer
        previousLayer_x_size = previous_layer_output.shape[0]
        previousLayer_y_size = previous_layer_output.shape[1]
        self.output_neurons = np.empty((previous_layer_output.shape[0]-self.pool_size[0]+1, previous_layer_output.shape[1]-self.pool_size[1]+1)) # new dimensions = previous dimensions -2
        self.xy_max_pairs
        # inputX and inputY are counters over input layer
        inputX = 0
        inputY = 0
        # iterate over the entire input matrix
        while(inputX+self.pool_size[0] < previousLayer_x_size+1):
            while(inputY+self.pool_size[1] < previousLayer_y_size+1):
                # filterX and filterY are the x and y positions going across the length of the imaginary pooling matrix
                # iterate over the whole imaginary pooling matrix
                maxValue = -1
                x_max = -1
                y_max = -1
                for filterX in range(0,self.pool_size[0]):
                    for filterY in range(0,self.pool_size[1]):
                        # find the max value from the current pool and set the releveant element of pool matrix to it
                        if( previous_layer_output[inputX+filterX][inputY+filterY] > maxValue ):
                            maxValue = previous_layer_output[inputX+filterX][inputY+filterY]
                            x_max = inputX + filterX
                            y_max = inputY + filterY
                self.output_neurons[inputX][inputY] = maxValue # set each element of pool matrix = max value
                self.xy_max_pairs.append((x_max,y_max)) # store where max element was so can find it easily in backpropogation
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
            self.weights[w] = (self.weights[w] *2) -1
        self.weights=self.weights.reshape(self.number_neurons,previous_layer_output_size)
        self.bias = rng.random(self.number_neurons)
        # He method
        # self.weights = rng.random(previous_layer_output_size*self.number_neurons)
        # self.bias = rng.random(self.number_neurons)
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
        
        # writeToFile("Hidden weights: " + str(self.weights))

    def get_weights(self):
        return self.weights

    def get_delta(self):
        return self.hidden_delta

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
            self.weights[w] = (self.weights[w] *2) -1
        self.weights=self.weights.reshape(self.number_neurons,previous_layer_output_size)
        self.bias = rng.random(self.number_neurons)
        # He method
        # self.weights = rng.random(previous_layer_output_size*self.number_neurons)
        # self.bias = rng.random(self.number_neurons)
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
        self.output_delta = []
        # calculate delta of output layer - how much they must change
        for i in range(len(output_errors)):
            self.output_delta.append(output_errors[i] * self.activation_derivative(self.output_neurons[i],self.activation_function))
        # update weights
        for i in range(len(self.weights)): # self.weights[i] = each output neuron
            for j in range(len(self.weights[i])): # self.weights[i][j] = output neuron's connection to each previous layer neuron
                self.weights[i][j] -= learning_rate * self.output_delta[i] * previous_layer_output[j]
            # update output biases
            self.bias[i] -= learning_rate * self.output_delta[i]
        # writeToFile("Output weights: " + str(self.weights))
            
    def get_delta(self):
        return self.output_delta

    def get_weights(self):
        return self.weights

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
                self.layers[i].backpropagation(self.layers[i+1].get_delta(),self.layers[i+1].get_xy_max_pairs(),input_data,learning_rate)
            elif(isinstance(self.layers[i], Input)):# last layer
                pass # no learnable paramaters at input so can finish backpropagation process now.

    def train(self, training_data_input, training_data_output, learning_rate, num_training_epochs):
        writeToFile("learning_rate: " + str(learning_rate))
        writeToFile("num_training_epochs: " + str(num_training_epochs))
        for epoch in range(num_training_epochs):
            epoch_start_time = time.time()
            total_error_at_output = 0
            # for each set of training data
            for training_data_num in range(len(training_data_input)):
                self.forward_propagation(training_data_input[training_data_num])
                total_error_at_output += sum( (np.array(training_data_output[training_data_num]) -np.array(self.layers[len(self.layers)-1].get_output()))**2)
                self.back_propagation(training_data_input[training_data_num], training_data_output[training_data_num],learning_rate)        
            epoch_end_time = time.time()
            print("Epoch: ", epoch, end=", ")
            print("Learning rate: ", learning_rate, end=", ")
            print("Error: ", total_error_at_output, end=", ")
            print(f"Runtime: {(epoch_end_time - epoch_start_time):.2f} seconds")
            self.epochErrors.append(total_error_at_output)
        plt.plot(range(len(self.epochErrors)), self.epochErrors)
        plt.xlabel("Num Epochs")
        plt.ylabel("Training Square Loss")
        plt.show()

        # Overfitting dataset checks
        correctPredictions = 0
        for i in range(len(training_data_input)):
            prediction = self.predict(training_data_input[i])
            print(prediction)
            if((isclose(prediction[0], training_data_output[i][0], abs_tol = 0.1)) and (isclose(prediction[1], training_data_output[i][1], abs_tol = 0.1))):
                correctPredictions += 1
            print("Expected: ", training_data_output[i], ", Got: ", prediction)
        print("Accuracy: ", correctPredictions/len(training_data_input))

### load training data ###
gestures = []
for i in range(46):
    if(i!=36):
        filepath='gesture_pics/gesture' + str(i) + '.npy'
        gestures.append(np.load(filepath))
training_data_input=[]
for i in range(9):
    newGestureData = []
    for x in range(len(gestures[i][0])):
        newGestureData.append(gestures[i][0][x]) # x
    for y in range(len(gestures[i][1])):
        newGestureData.append(gestures[i][1][y]) # y
    training_data_input.append(newGestureData)
# training_data_input = [[x1,x2...x9,y1,y2...y9]...]
# training_data_output = [[1,2,3,4,5,fist,peace,rockon,spock]]
training_data_output = [[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1],[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]]

### set up neural network ###
model = Network([
    Input(42, "sigmoid"),
    Hidden(100, "sigmoid"),
    Output(9, "sigmoid")])

### train network ###
model.train(training_data_input , training_data_output, learning_rate=0.1, num_training_epochs=6000)

details_file.close() # close network details textfile

# end program runtime and report duration
end_time = time.time() 
print(f"Program runtime: {(end_time - start_time):.2f} seconds")

# cleared delta = [] in each backprop. alg before appending to it
# changed weight initialization to only run first time network runs
# changed weight initliazation back to normal (not He)