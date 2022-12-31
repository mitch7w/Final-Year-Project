import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
from PIL import Image
import mediapipe
import cv2
import numpy as np

mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_hands = mediapipe.solutions.hands

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

    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

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

        vid = cv2.VideoCapture(0)  # define a video capture object
        ret, frame = vid.read()  # Capture the video frame by frame

        # mediapipe stuff
        image = frame
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                pinkyCoords = hand_landmarks.landmark[20]
                print("Pinky z: ", pinkyCoords.z)
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
            moveCube(pinkyCoords.x,pinkyCoords.y,-pinkyCoords.z)
            # print("x: ", pinkyCoords.x, ", y: ", pinkyCoords.y)
        Cube()  # display the cube

        pygame.display.flip()  # update the display
        # pygame.time.wait(10) # 10ms delay between canvas updates


main()
