import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import time
import math
start_time = time.time()

# random seed setting
rng = np.random.default_rng(1)

def sigmoid(x):
    z = np.exp(-x)
    return 1 / (1 + z)  

def sigmoid_derivative(x):
    return x * (1-x)

def forward_propagation(newInputLayer):
    global input_layer, hidden_layer, output_layer, hidden_bias, output_bias, weights_1, weights_2   
    input_layer = newInputLayer
    for neuron_number in range(len(hidden_layer)):
        # update hidden layer with dot product of inputs * weights all run through the activation function
        hidden_layer[neuron_number] = sigmoid(np.dot(newInputLayer , weights_1[neuron_number])+ hidden_bias[neuron_number])
    for neuron_number in range(len(output_layer)):
        # update output layer with dot product of hidden layer * weights all run through the activation function
        output_layer[neuron_number] = sigmoid(np.dot(hidden_layer , weights_2[neuron_number]) + output_bias[neuron_number])   

def predict(inValues):
    global output_layer
    forward_propagation(inValues)
    return output_layer

def back_propagation(desired_output):
    global input_layer, hidden_layer, output_layer, hidden_bias, output_bias, weights_1, weights_2   
    # calculate errors for output layer
    output_errors = output_layer - desired_output
    # calculate delta of output layer - how much they must change
    output_delta = []
    for i in range(len(output_errors)):
        output_delta.append(output_errors[i] * sigmoid_derivative(output_layer[i]))
    # calculate errors for hidden layer
    # for each neuron in hidden layer must have a error of its weights * neuron in next layer's delta
    hidden_errors = [0] * len(hidden_layer)
    for i in range(len(weights_2)): # weights_2[i] = each output neuron's weight array
        for j in range(len(weights_2[i])): # weights_2[i][j] = output neuron's connection to each hidden layer jth neuron
            # for each neuron in outputlayer, add to error the weight*output delta
            hidden_errors[j] += weights_2[i][j] * output_delta[i]
    hidden_delta = []
    for i in range(len(hidden_errors)):
        hidden_delta.append(hidden_errors[i] * sigmoid_derivative(hidden_layer[i]))
    # not sure if need below code
    # calculate errors for input layer
    input_errors = [0] * len(input_layer)
    for i in range(len(weights_1)): # weights_1[i] = each hidden neuron
        for j in range(len(weights_1[i])): # weights_1[i][j] = hidden neuron's connection to each input layer neuron
            input_errors[j] += weights_1[i][j] * hidden_delta[i]
    input_delta = []
    for i in range(len(input_errors)):
        input_delta.append(input_errors[i] * sigmoid_derivative(input_layer[i]))
    # update weights from input to hidden
    for i in range(len(weights_1)): # weights_1[i] = each hidden neuron
        for j in range(len(weights_1[i])): # weights_1[i][j] = hidden neuron's connection to each input layer neuron
            weights_1[i][j] -= learning_rate * hidden_delta[i] * input_layer[j]
        # update hidden biases
        hidden_bias[i] -= learning_rate * hidden_delta[i]
    # update weights from hidden to output
    for i in range(len(weights_2)): # weights_2[i] = each output neuron
        for j in range(len(weights_2[i])): # weights_2[i][j] = output neuron's connection to each hidden layer neuron
            weights_2[i][j] -= learning_rate * output_delta[i] * hidden_layer[j]
        # update output biases
        output_bias[i] -= learning_rate * output_delta[i]  

def train(training_data_input, training_data_output):
    global input_layer, hidden_layer, output_layer, hidden_bias, output_bias, weights_1, weights_2
    epochErrors = []
    for epoch in range(num_training_epochs):
        total_error_at_output = 0
        # for each set of training data
        for training_data_num in range(len(training_data_input)):
            forward_propagation(training_data_input[training_data_num])
            # calculate total error at output after each epoch
            # print("Expected output layer: ", training_data_output[training_data_num])
            # print("Actual output layer: ", output_layer)
            total_error_at_output += sum( (training_data_output[training_data_num] -output_layer)**2)
            back_propagation(training_data_output[training_data_num])
        print("Epoch: ", epoch, end=", ")
        print("Learning rate: ", learning_rate, end=", ")
        print("Error: ", total_error_at_output)
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

learning_rate = 0.1
num_training_epochs = 10000

training_data_input = [[2.7810836,2.550537003],
	[1.465489372,2.362125076],
	[3.396561688,4.400293529],
	[1.38807019,1.850220317],
	[3.06407232,3.005305973],
	[7.627531214,2.759262235],
	[5.332441248,2.088626775],
	[6.922596716,1.77106367],
	[8.675418651,-0.242068655],
	[7.673756466,3.508563011]]
# training_data_input = [[2.7810836,2.550537003]]
# training_data_output = [[0,1]]
training_data_output = [[0,1],[0,1],[0,1],[0,1],[0,1],[1,0],[1,0],[1,0],[1,0],[1,0]]


# initialize network
input_layer = np.empty(2) # input = 2 neurons
hidden_layer = np.empty(2) # hidden = 2 neurons
output_layer = np.empty(2) # output = 2 neurons
hidden_bias = np.zeros(2)
output_bias = np.zeros(2)
weights_1 = rng.random(4).reshape(2,2) # array of weight inputs for hidden_layer (6x6) = [hidden_neuron1_weights ... hidden_neuron6_weights]
weights_2 = rng.random(4).reshape(2,2) # array of weight inputs for output_layer (6x6) = [output_neuron1_weights...output_neuron2_weights]

# Train NN
train(training_data_input, training_data_output)

# Overfitting dataset checks
correctPredictions = 0
for i in range(len(training_data_input)):
    prediction = predict(training_data_input[i])
    if((math.isclose(prediction[0], training_data_output[i][0], abs_tol = 0.1)) and (math.isclose(prediction[1], training_data_output[i][1], abs_tol = 0.1))):
        correctPredictions += 1
    print("Expected: ", training_data_output[i], ", Got: ", prediction)
print("Accuracy: ", correctPredictions/len(training_data_input))

end_time = time.time()
print(f"Program runtime: {(end_time - start_time):.2f} seconds")