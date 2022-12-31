#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.transform import rescale
from scipy.signal import convolve2d
import freenect
import cv2
import importlib.util
spec1 = importlib.util.spec_from_file_location("frame_convert2", "/Users/mitch/Documents/University/Project/Project GitLab/prototypes/Kinect Basics/libfreenect/wrappers/python/frame_convert2.py")
frame_convert2 = importlib.util.module_from_spec(spec1)
spec1.loader.exec_module(frame_convert2)
from urllib.parse import parse_qsl
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
from PIL import Image

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
    (0,1,0),
    (0,0,1),
    (0,1,0),
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


# own coordinates for cube
cube_x_coord = 0 # max 8 = right of screen
cube_y_coord = 6 # max 6 = top of screen
cube_z_coord = 0
cube_length = 8/9

# API where can pass in 0-1 xy coords and have cube snap there
def moveCube(normalized_x_coord, normalized_y_coord, normalized_z_coord):
    global cube_x_coord, cube_y_coord, cube_z_coord, cube_length
    # 0 = left and top of screen
    # 1 = right and bottom of screen
    
    z_delta = normalized_z_coord - cube_z_coord
    cube_z_coord = normalized_z_coord
    # z coord requires shifting of x and y in order to stay in that place but appear to "shrink" or grow
    glTranslatef(0,0,z_delta)
    cube_length = 8/(9-cube_z_coord)
    glTranslatef(z_delta*0.5*cube_length, z_delta*-0.5*cube_length,z_delta)
    
    x_delta = normalized_x_coord - cube_x_coord
    y_delta = (normalized_y_coord - cube_y_coord)
    glTranslatef(x_delta*cube_length,y_delta*cube_length,0)
    cube_x_coord = normalized_x_coord
    cube_y_coord = normalized_y_coord
    
    print("x-coord: ", cube_x_coord,", ",end="")
    print("y-coord: ", cube_y_coord,", ",end="")
    print("z-coord: ", cube_z_coord)

def check_space_occupied(depth_data, proposed_x, proposed_y, proposed_z):
    # check depthInput at pixels to see if object there
    depth_x = int((proposed_x/8)*640)
    depth_y = int((proposed_y/6)*480)
    depth_z = int(650 -10*(proposed_z-3))
    for x in range(depth_x-20, depth_x+20):
        for y in range(depth_y-20, depth_y+20):
            if(depth_x >=0 and depth_x < 640 and depth_y >= 0 and depth_y < 480):
                if( depth_data[depth_x][depth_y] >= depth_z-50 and depth_data[depth_x][depth_y] <= depth_z+50):
                    return True
    return False

# start a pygame window and modify it for displaying OpenGL content
def main():
    pygame.init()
    display = (640,480)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) # clear OpenGL canvas

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0) # set perspective of OpenGL window
    # 45deg fov, aspect ratio, near + far clipping planes (where cube visible)
    glTranslatef(0.0,0.0, -10.0)
    glTranslatef(-4.0,0.0, 0)
    glTranslatef(0.0,2.7, 0)

    # main run loop
    while True:
        # get Kinect data
        depthInput = freenect.sync_get_depth()[0]
        videoInput = freenect.sync_get_video()[0]
        videoFeed = frame_convert2.video_cv(videoInput) # videoFeed is 480 x 640 x 3
        # videoFeed[0:100,500:600] access part of videoFeed like such
        # videoFeed[width][height] = image is 480 wide by 640 high
        # while True:
        #     cv2.imshow("video", videoFeed[200:300,400:500])
        #     cv2.waitKey(1)
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        texture_background = glGenTextures(1) # create OpenGL texture
        
        # get info of frame and convert to format needed for OpenGL texture
        background_image = cv2.flip(videoFeed,0)
        # image is 480 wide by 640 high - width left to right but height top to bottom
        background_image = cv2.rectangle(background_image, (200,400), (250,450), (0,0,255), 2) # rect coords are width x height
        background_image = Image.fromarray(background_image)
        image_width = background_image.size[0]
        image_height = background_image.size[1]
        background_image = background_image.tobytes('raw', 'BGRX', 0, -1)

            # create background texture
        glBindTexture(GL_TEXTURE_2D, texture_background)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexImage2D(GL_TEXTURE_2D, 0, 3, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, background_image)

        # draw background
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 1.0); glVertex3f( 1.0, -1.0, 0.0)
        glTexCoord2f(1.0, 0.0); glVertex3f( 1.0,  1.0, 0.0)
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0,  1.0, 0.0)
        glEnd()
        glDisable(GL_TEXTURE_2D)
        glPopMatrix()
        
        glDisable(GL_DEPTH_TEST)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: # upon close of window
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                # control cube with keyboard input
                if event.key == pygame.K_UP :
                    moveCube(cube_x_coord, cube_y_coord+1, cube_z_coord)
                if event.key == pygame.K_DOWN :
                    moveCube(cube_x_coord, cube_y_coord-1, cube_z_coord)
                if event.key == pygame.K_LEFT :
                    # glRotatef(1, 6, 0,0) # Rotate perspective by angle, x, y and z.
                    # object_collision = check_space_occupied(depthInput,cube_x_coord-1,cube_y_coord,-2)
                    # print("Object collision: ", object_collision)
                    # if(object_collision == False):
                    moveCube(cube_x_coord-1, cube_y_coord, cube_z_coord)
                if event.key == pygame.K_RIGHT :
                    # object_collision = check_space_occupied(depthInput,cube_x_coord+1,cube_y_coord,-2)
                    # print("Object collision: ", object_collision)
                    # if(object_collision == False):
                    moveCube(cube_x_coord+1, cube_y_coord, cube_z_coord)
                if event.key == pygame.K_x :
                    moveCube(4,3,-2)
                if event.key == pygame.K_i :
                    moveCube(cube_x_coord,cube_y_coord,cube_z_coord+1)
                if event.key == pygame.K_o :
                    moveCube(cube_x_coord,cube_y_coord,cube_z_coord-1)
        print("depthInput[middle]: ", depthInput[240][320], end="")
        print("x-coord: ", cube_x_coord,", ",end="")
        print("y-coord: ", cube_y_coord,", ",end="")
        print("z-coord: ", cube_z_coord)
        # at depth 650 cube z should be 3. at depth 700 cube z should be -2
        # formula relating z and depth_value is z = [ (650-d) / 10 ] + 3
        # should now hopefully be able to interact with real objects by checking next move-coord occupied with close to same z value as it?

        # glRotatef(1, 3, 1, 1) # Rotate perspective by angle, x, y and z.

        Cube() # display the cube

        pygame.display.flip() # update the display
        # pygame.time.wait(10) # 10ms delay between canvas updates

main()