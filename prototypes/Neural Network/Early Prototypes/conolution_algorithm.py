import numpy as np

def convolution(input, filter):
    # performs convolution with an input matrix and filter matrix
    # filter and input matrix dimensions
    filter_x_size = filter.shape[0]
    filter_y_size = filter.shape[1]
    input_x_size = input.shape[0]
    input_y_size = input.shape[1]
    # set output matrix size - no strides or padding
    newOutput = np.empty([input_x_size-filter_x_size+1, input_y_size-filter_y_size+1],dtype=np.int64) 
    # inputX and inputY are counters specifying where the top left corner of the currently considered input matrix is
    inputX = 0
    inputY = 0
    # iterate over the entire input matrix
    while(inputX+filter_x_size < input_x_size+1):
        while(inputY+filter_y_size < input_y_size+1):
            # filterX and filterY are the x and y positions going across the filter matrix
            # iterate over the whole filter matrix
            sumMultiplications = 0 # add all the multiplications together
            for filterX in range(0,filter_x_size):
                for filterY in range(0,filter_y_size):
                    # print("Multiplying input", input[inputX+filterX][inputY+filterY], "with filter", filter[filterX][filterY])
                    # each element of current subsection of input matrix * each element of filter matrix
                    sumMultiplications += input[inputX+filterX][inputY+filterY] * filter[filterX][filterY]
            newOutput[inputX][inputY] = sumMultiplications # set each element of output matrix = sum of multiplications
            inputY +=1
        inputX +=1
        inputY = 0
    return newOutput

in1 = np.array([[1,2,3],[5,6,7],[9,10,11]])
f1 = np.array([[0,0],[0,1]])

convolveOut = convolution(in1,f1)
print(convolveOut)