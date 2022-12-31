import cv2
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
    ret, frame = vid.read() # Capture the video frame by frame
    # get info of frame and convert to format needed for OpenGL texture
    texture_frame = Image.fromarray(frame)
    image_width = texture_frame.size[0]
    image_height = texture_frame.size[1]
    texture_frame = texture_frame.tobytes('raw', 'BGRX', 0, -1)

    # create OpenGL texture
    print("hello")
    texture_ID = glGenTextures(1)
    
    glBindTexture(GL_TEXTURE_2D, texture_ID)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_ID)
    glBindTexture(GL_TEXTURE_2D, texture_ID)
    glBegin(GL_POLYGON)
    glTexCoord2fv([0.0, 0.0])
    glVertex3fv([1.1, 1.1, -1.1])
    glTexCoord2fv([1.0, 0.0])
    glVertex3fv([0.0, 1.1, -1.1])
    glTexCoord2fv([1.0, 1.0])
    glVertex3fv([0.0, 1.1, 0.0])
    glTexCoord2fv([0.0, 1.0])
    glVertex3fv([1.1, 1.1, 0.0])
    glEnd()

    cv2.imshow('frame', frame) # Display the resulting frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()
cv2.destroyAllWindows()
