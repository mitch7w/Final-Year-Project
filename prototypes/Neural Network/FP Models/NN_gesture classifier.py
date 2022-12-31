import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation 
import time
import math
import mediapipe
import cv2

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
num_training_epochs = 500
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


# initialize network
input_layer = np.empty(42)
hidden_layer = np.empty(100)
output_layer = np.empty(9) # output = 9 neurons
hidden_bias = np.zeros(100)
output_bias = np.zeros(9)
weights_1 = rng.random(100*42)
for w in range(len(weights_1)):
    weights_1[w] = (weights_1[w] *2) -1
weights_1=weights_1.reshape(100,42)
weights_2=rng.random(9*100)
for w in range(len(weights_2)):
    weights_2[w] = (weights_2[w] *2) -1
weights_2=weights_2.reshape(9,100)

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

# now test with live images

mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_hands = mediapipe.solutions.hands

cap = cv2.VideoCapture(0)
pinkyCoords = {}
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            continue

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        handCoords=[]
        predictionText="Prediction: "
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                handCoords = hand_landmarks.landmark
                x = []
                y=[]
                z=[]
                for i in range(21):
                    x.append(handCoords[i].x)
                    y.append(handCoords[i].y)
                    z.append(handCoords[i].z)
                handCoords = []
                for xrange in range(len(x)):
                    handCoords.append(x[xrange]) # x
                for yrange in range(len(y)):
                    handCoords.append(y[yrange]) # y
                oLay = predict(handCoords)
                # find largest output predictor (softmax)
                largestOutIndex=0
                for i in range(1,9):
                    if(oLay[i]>oLay[largestOutIndex]):
                        largestOutIndex = i
                outGestures=["One","Two","Three","Four","Five","Fist","Peace","Rockon","OK"]
                predictionText = "Predicted gesture: " +str(outGestures[largestOutIndex])
                print(predictionText)
        cv2.putText(image,predictionText,(10,500),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,2)
        cv2.imshow('Predicted Gesture', image)
        if cv2.waitKey(5) == ord(" "):
            cv2.imwrite("predictedGestureTest1.jpg", image)
cap.release()

end_time = time.time()
print(f"Program runtime: {(end_time - start_time):.2f} seconds")