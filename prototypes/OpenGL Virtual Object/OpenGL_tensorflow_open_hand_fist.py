# -*- coding: utf-8 -*-

# Setup
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import scipy.signal
import cv2
from numpy.random import default_rng
import random
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import cv2
import numpy as np
# set random seed for random number generator
rng = default_rng(777)

def loadImage(imageName):
    if(isinstance(imageName,str)):
        # load image if just it's name was passed in
        imageName = cv2.imread(imageName,1) 
    # imageName is now the RGB-passed-in image
    # perform skin segmentation using the Ycbcr color space
    # put in tracker code here
    image = cv2.resize(imageName, (320,240), interpolation = cv2.INTER_AREA)
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
    for r_counter in range(0,320,5):
        for c_counter in range(0,240,5):
            region_sum = np.sum(mask[r_counter:r_counter+60,c_counter:c_counter+80]) # sum all white pixels in proposal region
            if(region_sum > 100000): # if more white pixels here than average
                south_delta = abs(r_counter-240)
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
                if( abs(south_deltas[current_region] - southmost_delta) < 50):
                    largest_south_index = current_region
        crop_coords = [c_region_counters[largest_south_index],c_region_counters[largest_south_index],r_region_counters[largest_south_index],r_region_counters[largest_south_index]]
        if(crop_coords[0]-15 <0): # how much extra left to do
            crop_coords[0] = 0
        else:
            crop_coords[0] -= 15

        if(crop_coords[1]+100 >319): # how much extra right to do
            crop_coords[1] = 319
        else:
            crop_coords[1] += 100

        if(crop_coords[2] - 40 < 0): # how much extra up to do
            crop_coords[2] = 0
        else:
            crop_coords[2] -= 40

        if(crop_coords[3]+80 >239): # how much extra down to do
            crop_coords[3] = 239
        else:
            crop_coords[3] += 80
    else:
        print("Error!: no regions with pixel concentration above threshold")
        crop_coords = [c_region_counters[largest_south_index],c_region_counters[largest_south_index]+115,r_region_counters[largest_south_index],r_region_counters[largest_south_index]+120]
    extracted_hand = result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]] 
    if(extracted_hand.shape != (120, 115, 3)):
        extracted_hand = cv2.resize(result[crop_coords[2]:crop_coords[3],crop_coords[0]:crop_coords[1]], (115,120), interpolation = cv2.INTER_AREA)
    # print("extracted_hand.shape: ", extracted_hand.shape)
    # make sure all images are same size even if cropped on the side
    loadedImage = np.array(extracted_hand)/255 # divide by 255 in order to normalize input
    if(loadedImage.shape[2] !=3): # if it's not RGB put it in a 3D shape
        new3DImage = np.empty((loadedImage.shape[0],loadedImage.shape[1],1))
        for x in range(len(new3DImage)):
            for y in range(len(new3DImage[x])):
                for i in range(len(new3DImage[x][y])):
                    new3DImage[x][y][i] = loadedImage[x][y]
        loadedImage = new3DImage
    return np.array(loadedImage)

# load images
training_data_input = []
for i in range(1,241):
    imageName = "/Users/mitch/Documents/University/Project/hand_detector_fist_open_hand/gesture" + str(i) + ".jpg"
    training_data_input.append(loadImage(imageName))
print("training_data_input shape: ", np.array(training_data_input).shape)

# load output classes
training_data_output = []
for j in range(2):
    for i in range(60):
        training_data_output.append([1,0]) # Open
    for i in range(60):
        training_data_output.append([0,1]) # Fist
print("training_data_output shape: ", np.array(training_data_output).shape)

# shuffle training data
new_list =list(zip(training_data_input, training_data_output))
random.shuffle(new_list)
training_data_input, training_data_output = zip(*new_list)
print("All training data set up.")

# separate training and testing data

x_train = np.array(training_data_input[:-24])
y_train = np.array(training_data_output[:-24])
x_test = np.array(training_data_input[-24:])
y_test = np.array(training_data_output[-24:])
print("x_train.shape: ", x_train.shape)

num_output_classes = 2
input_shape = training_data_input[0].shape
print(input_shape)

# Build the model
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
        layers.MaxPooling2D(pool_size=(3,3),strides=(2,2)),
        layers.Flatten(),
        layers.Dense(1000, activation="relu"),
        layers.Dense(num_output_classes, activation="softmax"), 
    ]
)

model.summary()

#Train the model
batch_size = 1
epochs = 2

# Setup model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Stop the training if accuracy of 1 is reached
callback = callbacks.EarlyStopping(monitor='accuracy', patience=1)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])


################################################################################################
################################################################################################
################################################################################################
################################################################################################
##############################################################################################
# Use model to move objects with OpenGL

# create the cube with the defined vertices and edges
def Cube():
    vertices = (
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
        (0, 1),
        (0, 3),
        (0, 4),
        (2, 1),
        (2, 3),
        (2, 7),
        (6, 3),
        (6, 4),
        (6, 7),
        (5, 1),
        (5, 4),
        (5, 7)
    )
    colors = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (0, 1, 0),
        (1, 1, 1),
        (0, 1, 1),
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 0),
        (1, 1, 1),
        (0, 1, 1),
    )
    surfaces = (
        (0, 1, 2, 3),
        (3, 2, 7, 6),
        (6, 7, 5, 4),
        (4, 5, 1, 0),
        (1, 5, 7, 2),
        (4, 0, 3, 6)
    )
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x += 1
            # color each surface with a slightly different color
            glColor3fv(colors[x])
            glVertex3fv(vertices[vertex])
    glEnd()
    # glBegin denotes start of special OpenGL commands to follow
    glBegin(GL_LINES)
    for edge in edges:
        for vertex in edge:
            glVertex3fv(vertices[vertex])
    glEnd()  # end of special OpenGL commands

# own coordinates for cube
cube_x_coord = 0 # max 8 = right of screen
cube_y_coord = 0 # max 6 = bottom of screen
cube_z_coord = 0

# API where can pass in 0-1 xy coords and have cube snap there
def moveCube(normalized_x_coord, normalized_y_coord, normalized_z_coord):
    global cube_x_coord, cube_y_coord, cube_z_coord
    # 0 = left and top of screen
    # 1 = right and bottom of screen
    x_delta = (normalized_x_coord*8.0) - cube_x_coord
    y_delta = -((normalized_y_coord*6.0) - cube_y_coord)
    z_delta = (normalized_z_coord)- cube_z_coord
    cube_x_coord = normalized_x_coord*8.0
    cube_y_coord = normalized_y_coord*6.0
    glTranslatef(x_delta,y_delta, 0)
    # z coord requires shifting of x and y in order to stay in that place but appear to "shrink" or grow
    # glTranslatef(z_delta*0.5, z_delta*-0.5,z_delta)
    # cube_z_coord = normalized_z_coord

# start a pygame window and modify it for displaying OpenGL content

def handTracking(image):
    image = cv2.resize(image, (320,240), interpolation = cv2.INTER_AREA)
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
    for r_counter in range(0,320,5):
        for c_counter in range(0,240,5):
            region_sum = np.sum(mask[r_counter:r_counter+60,c_counter:c_counter+80]) # sum all white pixels in proposal region
            if(region_sum > 100000): # if more white pixels here than average
                south_delta = abs(r_counter-240)
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
                if( abs(south_deltas[current_region] - southmost_delta) < 50):
                    largest_south_index = current_region
    return c_region_counters[largest_south_index] , r_region_counters[largest_south_index]

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # clear OpenGL canvas

    # set perspective of OpenGL window
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
    # 45deg fov, aspect ratio, near + far clipping planes (where cube visible)
    glTranslatef(0.0,0.0, -10.0) # move the perspective in the z-plane by -5
    glTranslatef(-4.0,0.0, 0) # move the perspective in the z-plane by -5
    glTranslatef(0.0,3.0, 0) # move the perspective in the z-plane by -5

    pinkyCoords = {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0
    }
    noHandsYet = True

    # main run loop
    while True:
        # display image

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        texture_background = glGenTextures(1)  # create OpenGL texture

        vid = cv2.VideoCapture(1)  # define a video capture object
        ret, frame = vid.read()  # Capture the video frame by frame

        # prediction
        loaded_image_new = np.array(loadImage(frame)).reshape(1,input_shape[0],input_shape[1],input_shape[2])
        new_prediction = model.predict(loaded_image_new)
        gesture_command = "none"
        print("new_prediction: ", new_prediction,end="")
        if(new_prediction[0][0]>new_prediction[0][1]): # softmax layer basically
            gesture_command = "openhand"
            print("open hand")
        else:
            gesture_command = "fist"
            print("fist")
        noHandsYet = False

        # get info of frame and convert to format needed for OpenGL texture
        background_image = cv2.flip(frame, 0)
        background_image = Image.fromarray(background_image)
        image_width = background_image.size[0]
        image_height = background_image.size[1]
        background_image = background_image.tobytes('raw', 'BGRX', 0, -1)

        # create background texture
        glBindTexture(GL_TEXTURE_2D, texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, image_width, image_height,
                     0, GL_RGBA, GL_UNSIGNED_BYTE, background_image)

        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0)
        glVertex3f(-1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 1.0)
        glVertex3f(1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 0.0)
        glVertex3f(1.0,  1.0, 0.0)
        glTexCoord2f(0.0, 0.0)
        glVertex3f(-1.0,  1.0, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()

        glDisable(GL_DEPTH_TEST)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # upon close of window
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # control cube with keyboard input
                if event.key == pygame.K_LEFT:
                    # Rotate perspective by angle, x, y and z.
                    glRotatef(1, 6, 0, 0)
                if event.key == pygame.K_RIGHT:
                    # Rotate perspective by angle, x, y and z.
                    glRotatef(1, 0, 6, 0)
                if event.key == pygame.K_UP:
                    # Rotate perspective by angle, x, y and z.
                    glRotatef(1, 0, 0, 6)
                if event.key == pygame.K_DOWN:
                    # Rotate perspective by angle, x, y and z.
                    glRotatef(1, 0, 0, -6)
                if event.key == pygame.K_i:
                    glTranslatef(0.0, 0.0, 0.1)
                if event.key == pygame.K_o:
                    glTranslatef(0.0, 0.0, -0.1)
        # glRotatef(1, 3, 1, 1) # Rotate perspective by angle, x, y and z.
        if(noHandsYet):
            glTranslatef(0.0,0.0,0.0)
        else:
            # if(gesture_command=="openhand"):
                # moveCube(0.3,0.3,0.3)
            if(gesture_command=="fist"):
                x_hand, y_hand = handTracking(frame)
                moveCube(x_hand/240,y_hand/320,0.6)
                
        Cube()  # display the cube

        pygame.display.flip()  # update the display
        # pygame.time.wait(10) # 10ms delay between canvas updates


main()

