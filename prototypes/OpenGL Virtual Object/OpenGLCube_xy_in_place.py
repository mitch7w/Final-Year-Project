from urllib.parse import parse_qsl
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import cv2
from PIL import Image

# own coordinates for cube
cube_x_coord = 0 # max 8 = right of screen
cube_y_coord = 0 # max 6 = bottom of screen
cube_z_coord = 0
cube_scale = 1

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
    (1,0,1),
    (1,0,0),
    (1,1,0),
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

# start a pygame window and modify it for displaying OpenGL content
def main():
    global cube_x_coord, cube_y_coord, cube_z_coord, cube_scale
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT) # clear OpenGL canvas
    gluPerspective(45, (display[0]/display[1]), 0.1, 50.0) # set perspective of OpenGL window
    # 45deg fov, aspect ratio, near + far clipping planes (where cube visible)
    glTranslatef(-3.97,-2.7, -10.0) # move the cube to bottom left

    # main run loop
    while True:
        # display image

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        texture_background = glGenTextures(1) # create OpenGL texture
        
        vid = cv2.VideoCapture(0) # define a video capture object
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

        # draw background2
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
                if event.key == pygame.K_LEFT :
                    glTranslatef(-1, 0, 0)
                    cube_x_coord -= 1*cube_scale*0.5
                if event.key == pygame.K_RIGHT :
                    glTranslatef(1, 0, 0)
                    cube_x_coord += 1*cube_scale*0.5
                if event.key == pygame.K_UP :
                    glTranslatef(0, 1, 0)
                    cube_y_coord += 1*cube_scale
                if event.key == pygame.K_DOWN :
                    glTranslatef(0, -1, 0)
                    cube_y_coord -= 1*cube_scale
                if event.key == pygame.K_i :
                    # glScalef(2,2,2)
                    glTranslatef(0,0, 1)
                    glTranslatef(0.4,0.4, 0)
                    cube_scale += 1
                if event.key == pygame.K_o :
                    # glScalef(0.5,0.5,0.5)
                    glTranslatef(0,0, -1)
                    cube_scale -= 1
                if event.key == pygame.K_w :
                    glRotatef(45,1,0,0) # Rotate perspective by angle, x, y and z.
                if event.key == pygame.K_s :
                    glRotatef(45,-1,0,0) # Rotate perspective by angle, x, y and z.
                if event.key == pygame.K_a :
                    glRotatef(45,0,1,0) # Rotate perspective by angle, x, y and z.
                if event.key == pygame.K_d :
                    glRotatef(45,0,-1,0) # Rotate perspective by angle, x, y and z.
                
        Cube() # display the cube    
        print("X: ", cube_x_coord, ", Y: ", cube_y_coord, ", Z: ", cube_z_coord, ", Scale: ", cube_scale)
        pygame.display.flip() # update the display
    
main()