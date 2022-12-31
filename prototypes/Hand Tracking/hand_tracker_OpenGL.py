import cv2
import numpy as np
import sys
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
# makes arrays print out in full
np.set_printoptions(threshold=sys.maxsize) 

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
def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) # clear OpenGL canvas

    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0) # set perspective of OpenGL window
    # 45deg fov, aspect ratio, near + far clipping planes (where cube visible)
    glTranslatef(0.0,0.0, -10.0) # move the perspective in the z-plane by -5
    glTranslatef(-4.0,0.0, 0) # move the perspective in the z-plane by -5
    glTranslatef(0.0,3.0, 0) # move the perspective in the z-plane by -5

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
        ret, frame = vid.read() # Capture the video frame by frame
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
                if event.key == pygame.K_LEFT :
                    glRotatef(1, 6, 0,0) # Rotate perspective by angle, x, y and z.
                if event.key == pygame.K_RIGHT :
                    glRotatef(1, 0,6,0) # Rotate perspective by angle, x, y and z.
                if event.key == pygame.K_UP :
                    glRotatef(1, 0,0,6) # Rotate perspective by angle, x, y and z.
                if event.key == pygame.K_DOWN :
                    glRotatef(1, 0,0,-6) # Rotate perspective by angle, x, y and z.
                if event.key == pygame.K_x :
                    moveCube(4,4,1)
                if event.key == pygame.K_z :
                    moveCube(6,6,2)
        # hand tracker
        image = cv2.resize(frame, (320,240), interpolation = cv2.INTER_AREA)
        image_ycrcb = np.array(cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB))
        # print(image_ycrcb[0][0])
        skin1 = (50, 89, 136)
        skin2 = (231, 147, 181)
        mask = cv2.inRange(image_ycrcb, skin1, skin2)
        result = cv2.bitwise_and(image, image, mask=mask)
        # cv2.imshow("Color", result)
        # cv2.imshow("Mask", mask) # 240 x 320 white or black image

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
        if(len(south_deltas) ==0):
            continue
        southmost_delta = np.amin(south_deltas)
        # find largest region within 50px of southmost point
        largest_south_index = 0
        for current_region in range(len(region_sums)):
            # print("south_deltas[current_region] - southmost_delta: ", south_deltas[current_region] - southmost_delta)
            if(region_sums[current_region] > region_sums[largest_south_index]):
                if( abs(south_deltas[current_region] - southmost_delta) < 50):
                    largest_south_index = current_region
        moveCube((c_region_counters[largest_south_index]+10)/320, (r_region_counters[largest_south_index]+20)/240, 0)
        # moveCube(1,1,0)
        Cube() # display the cube

        pygame.display.flip() # update the display
        # pygame.time.wait(10) # 10ms delay between canvas updates

main()