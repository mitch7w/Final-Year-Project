# # # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # import scipy.signal

# # # # # # # # # # # # # # # # # # # # # def fullConvolution(input,filter):
# # # # # # # # # # # # # # # # # # # # #     # Can use regular convolution method so long as right amount padding added
# # # # # # # # # # # # # # # # # # # # #     filterLength = filter.shape[0]
# # # # # # # # # # # # # # # # # # # # #     paddedInputLength = (3*filterLength) -2
# # # # # # # # # # # # # # # # # # # # #     # create new version of input with all padding
# # # # # # # # # # # # # # # # # # # # #     newInput = np.zeros(paddedInputLength**2).reshape((paddedInputLength,paddedInputLength))
# # # # # # # # # # # # # # # # # # # # #     xCounter = filterLength-1
# # # # # # # # # # # # # # # # # # # # #     yCounter = filterLength-1
# # # # # # # # # # # # # # # # # # # # #     for xCounter in range(filterLength-1,(2*filterLength)-1):
# # # # # # # # # # # # # # # # # # # # #         for yCounter in range(filterLength-1,(2*filterLength)-1):
# # # # # # # # # # # # # # # # # # # # #             newInput[xCounter][yCounter] = input[xCounter-filterLength-1][yCounter-filterLength-1] # put in input values that aren't zero
# # # # # # # # # # # # # # # # # # # # #     print(newInput)

# # # # # # # # # # # # # # # # # # # # # i1 = np.array([[1,1,1],[1,1,1],[1,1,1]])
# # # # # # # # # # # # # # # # # # # # # f1 = np.array([[2,2],[2,2]])

# # # # # # # # # # # # # # # # # # # # # fullConvolution = scipy.signal.convolve2d(i1,f1, "full", "fill",0)
# # # # # # # # # # # # # # # # # # # # # print(fullConvolution)

# # # # # # # # # # # # # # # # # # # # # expectedOutput = np.arange(1,200)
# # # # # # # # # # # # # # # # # # # # # print(expectedOutput)

# # # # # # # # # # # # # # # # # # # # hello = [1,2,3,4]
# # # # # # # # # # # # # # # # # # # # hello.reverse()
# # # # # # # # # # # # # # # # # # # # hello = np.array(hello).reshape((2,2))
# # # # # # # # # # # # # # # # # # # # print(hello)

# # # # # # # # # # # # # # # # # # # # test = expectedOutput = np.arange(0,1,0.005)
# # # # # # # # # # # # # # # # # # # # print(test)

# # # # # # # # # # # # # # # # # # # # multi = np.array([[2,3],[4,5]])
# # # # # # # # # # # # # # # # # # # # multi = multi/2
# # # # # # # # # # # # # # # # # # # # print(multi)

# # # # # # # # # # # # # # # # # # # # testOutput = [1.60716336e-03, 7.91936456e-03, 8.26471139e-06, 4.01867718e-04,
# # # # # # # # # # # # # # # # # # # #  9.35690716e-02, 1.70163495e-01, 1.24027495e-01, 1.77563037e-01,
# # # # # # # # # # # # # # # # # # # #  2.45693162e-01, 4.78340052e-15, 1.02063268e-01, 3.19588394e-01,
# # # # # # # # # # # # # # # # # # # #  1.61013018e-01, 3.49145252e-01, 3.36877894e-01, 3.80395404e-01,
# # # # # # # # # # # # # # # # # # # #  3.81811216e-01, 4.18900268e-01, 4.62484843e-01, 5.01323815e-01,
# # # # # # # # # # # # # # # # # # # #  4.66938674e-01, 5.17117713e-01, 5.34452336e-01, 5.45802734e-01,
# # # # # # # # # # # # # # # # # # # #  6.67801533e-01, 8.69141085e-01, 6.81837871e-01, 7.21171934e-01,
# # # # # # # # # # # # # # # # # # # #  6.84023761e-01, 7.03753894e-01, 7.82855790e-01, 8.12116800e-01,
# # # # # # # # # # # # # # # # # # # #  7.94133322e-01, 9.99998046e-01, 7.30766604e-01, 9.77094373e-01,
# # # # # # # # # # # # # # # # # # # #  9.30617533e-01, 8.55184335e-01, 9.70416822e-01, 1.00000000e+00]
# # # # # # # # # # # # # # # # # # # # for i in testOutput:
# # # # # # # # # # # # # # # # # # # #     print(i)

# # # # # # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # # # # # # from pathlib import Path

# # # # # # # # # # # # # # # # # # # # path = Path('~/data/tmp/').expanduser()
# # # # # # # # # # # # # # # # # # # # path.mkdir(parents=True, exist_ok=True)

# # # # # # # # # # # # # # # # # # # x=[1,2,3,4,5,6,7]

# # # # # # # # # # # # # # # # # # # np.save('Trained Nets/MNIST_classifier_weights1', x)

# # # # # # # # # # # # # # # # # # # # x_loaded = np.load('x.npy')

# # # # # # # # # # # # # # # # # # class Network:
# # # # # # # # # # # # # # # # # #     def __init__(self,number):
# # # # # # # # # # # # # # # # # #         self.number = number

# # # # # # # # # # # # # # # # # # hello= Network(1)

# # # # # # # # # # # # # # # # # # print(isinstance(hello, Network))

# # # # # # # # # # # # # # # # # for i in range(4-1,-1,-1):
# # # # # # # # # # # # # # # # #     print(i)

# # # # # # # # # # # # # # # # a = [1,2,3]
# # # # # # # # # # # # # # # # b = [4,6,11]
# # # # # # # # # # # # # # # # c = b-a
# # # # # # # # # # # # # # # layers = [1,2,3,4]
# # # # # # # # # # # # # # # for i in range(len(layers)-1,-1,-1):
# # # # # # # # # # # # # # #     print(i)
# # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # hello = np.array([[1,2,3],[4,5,6]])
# # # # # # # # # # # # # # # for x in range(len(hello)):
# # # # # # # # # # # # # # #     hello[x] = hello[x][::-1]
# # # # # # # # # # # # # # # hello = hello[::-1]
# # # # # # # # # # # # # # # print(hello)

# # # # # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # # # # hello = np.empty((2,3,2))
# # # # # # # # # # # # # # # hello[0][0] = (4,5)
# # # # # # # # # # # # # # # print(hello)

# # # # # # # # # # # # # # import scipy.signal
# # # # # # # # # # # # # # input = [[1,2],[3,4]]
# # # # # # # # # # # # # # filter = [[1,1],[1,1]]
# # # # # # # # # # # # # # result = scipy.signal.convolve2d(input,filter, "full", "fill",0)
# # # # # # # # # # # # # # print(result)

# # # # # # # # # # # # # # import random
# # # # # # # # # # # # # # list1 = [1,2,3,4]
# # # # # # # # # # # # # # list2 = ['a','b','c','d']
# # # # # # # # # # # # # # list3 =list(zip(list1, list2))
# # # # # # # # # # # # # # random.shuffle(list3)
# # # # # # # # # # # # # # list1, list2 = zip(*list3)
# # # # # # # # # # # # # # print(list1)
# # # # # # # # # # # # # # print(list2)
# # # # # # # # # # # # # from math import sqrt
# # # # # # # # # # # # # from numpy.random import default_rng
# # # # # # # # # # # # # rng = default_rng(777)
# # # # # # # # # # # # # weights = rng.random(100*4)
# # # # # # # # # # # # # standard_deviation = sqrt(2/100)
# # # # # # # # # # # # # for w in range(len(weights)):
# # # # # # # # # # # # #     weights[w] = weights[w] * standard_deviation
# # # # # # # # # # # # # weights=weights.reshape(4,100)
# # # # # # # # # # # # # print(weights)

# # # # # # # # # # # # import numpy as np
# # # # # # # # # # # # array1 = np.array([[[1,2,3],[4,5,6]],[[[7,8,9],[10,11,12]]]])
# # # # # # # # # # # # print(array1)
# # # # # # # # # # # # result = np.where((array1[::2] >2), array1, 0)
# # # # # # # # # # # # result = np.where((result[::2] <5), result, 0)
# # # # # # # # # # # # # result = np.where(result < 5 , result, 0)
# # # # # # # # # # # # print(result)

# # # # # # # # # # # import numpy as np
# # # # # # # # # # # values = [1,2,3,4,5,-2]
# # # # # # # # # # # index_min = np.argmax(values)
# # # # # # # # # # # print(index_min)

# # # # # # # # # # for r_counter in range(0,240,120):
# # # # # # # # # #     print(r_counter)

# # # # # # # # # A = ["r", "s", "t", "u", "v", "w", "x", "y", "z"]
# # # # # # # # # B = [ 1,   2,   3,    7,   5,   5,   22,   4,   6]
# # # # # # # # # C = ["hello","hello","hello","hello","hello","hello","bye","hello","hello"]
# # # # # # # # # result_list = [i for _,i in sorted(zip(B,A,C))]
# # # # # # # # # print(result_list)

# # # # # # # # import numpy as np
# # # # # # # # x = np.array([[1,2,3],[4,5,6]])
# # # # # # # # y = x + 1
# # # # # # # # print(y)
# # # # # # # # # y = [[3,4,5],[6,7,8]]
# # # # # # # # # w = [[3,3,3],[4,4,4]]
# # # # # # # # # z = sum(x,y,w)
# # # # # # # # # # z = np.add(x,y)
# # # # # # # # # print(z)

# # # # # # # # print(int(3.9))

# # # # # # # # from vpython import *
# # # # # # # # sphere()

# # # # # # # from OpenGL.GL import *
# # # # # # # from OpenGL.GLUT import *
# # # # # # # from OpenGL.GLU import *

# # # # # # # w,h= 500,500
# # # # # # # def square():
# # # # # # #     glBegin(GL_QUADS)
# # # # # # #     glVertex2f(100, 100)
# # # # # # #     glVertex2f(200, 100)
# # # # # # #     glVertex2f(200, 200)
# # # # # # #     glVertex2f(100, 200)
# # # # # # #     glEnd()

# # # # # # # def iterate():
# # # # # # #     glViewport(0, 0, 500, 500)
# # # # # # #     glMatrixMode(GL_PROJECTION)
# # # # # # #     glLoadIdentity()
# # # # # # #     glOrtho(0.0, 500, 0.0, 500, 0.0, 1.0)
# # # # # # #     glMatrixMode (GL_MODELVIEW)
# # # # # # #     glLoadIdentity()

# # # # # # # def showScreen():
# # # # # # #     glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
# # # # # # #     glLoadIdentity()
# # # # # # #     iterate()
# # # # # # #     glColor3f(1.0, 0.0, 3.0)
# # # # # # #     square()
# # # # # # #     glutSwapBuffers()

# # # # # # # glutInit()
# # # # # # # glutInitDisplayMode(GLUT_RGBA)
# # # # # # # glutInitWindowSize(500, 500)
# # # # # # # glutInitWindowPosition(0, 0)
# # # # # # # wind = glutCreateWindow("OpenGL Coding Practice")
# # # # # # # glutDisplayFunc(showScreen)
# # # # # # # glutIdleFunc(showScreen)
# # # # # # # glutMainLoop()

# # # # # # import numpy as np
# # # # # # hello = np.array([[1,2,3,4,5],[6,7,8,9,20],[11,12,13,14,15]])
# # # # # # print(np.max(hello[0:3,1:2]))

# # # # # import numpy as np
# # # # # import skimage.measure

# # # # # a = np.array([
# # # # #       [  20,  200,   -5,   23],
# # # # #       [ -13,  134,  119,  100],
# # # # #       [ 120,   32,   49,   25],
# # # # #       [-120,   12,    9,   23]
# # # # # ])
# # # # # print(a[0:2,0:3])

# # # # # x = np.array([[1,2,3,-4],[5,6,7,8]])
# # # # # print( x * (x > 0))

# # # # import numpy as np
# # # # def activation(output_array, activation_function):
# # # #         if(activation_function=="softmax"):
# # # #             exponents_output_array = np.exp(output_array)
# # # #             return exponents_output_array / np.sum(exponents_output_array)

# # # # data = [1, 3, 2]
# # # # # convert list of numbers to a list of probabilities
# # # # result = activation(data,"softmax")
# # # # # report the probabilities
# # # # print(result)

# # # import numpy as np
# # # mat = np.array([[  20,  200,   -5,   23,1,2],
# # #                 [ -13,  134,  119,  100,2,3],
# # #                 [ 120,   32,   49,   25,3,4],
# # #                 [-120,   12,   9,   23,4,5],
# # #                 [-122,   13,   10,   24,5,6],
# # #                 [-126,   14,   11,   25,6,7]])

# # # M, N = mat.shape
# # # K = 3
# # # L = 2

# # # MK = M // K
# # # NL = N // L
# # # print(mat[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3)))

# # import numpy as np
# # hello = np.array([[1,2,3],
# #         [4,5,6,-1]])
# # print(np.argmin(hello[1]))

# test = [["x",3],["y",4]]
# test.reverse()
# print(test)

import numpy as np
matrix = np.arange(32*36*16).reshape(32,36,16)
print(matrix)
stride_size = 2
kernel_size = (3,3)
strided_pool_groups = np.lib.stride_tricks.sliding_window_view(matrix, kernel_size)[::stride_size,::stride_size]
strided_pool_groups_flattened = strided_pool_groups.reshape((strided_pool_groups.shape[0]*strided_pool_groups.shape[1],strided_pool_groups.shape[2]*strided_pool_groups.shape[3]))
maxpooled = np.max(strided_pool_groups_flattened, axis=1)

print("strided_pool_groups: ", strided_pool_groups)
print(strided_pool_groups.shape)
print("strided_pool_groups_flattened: ", strided_pool_groups_flattened )
print("maxpooled: ", maxpooled)