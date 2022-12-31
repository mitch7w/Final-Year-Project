import cv2
import numpy as np

top_left_coordinates = [50,50]
width_cube = 50
degrees_rotated = 0
# BGR colors
RED = (0, 0, 255)
GREEN = (0,255,0)
BLUE = (255,0,0)
ORANGE = (34,156,228)

while True:
    degrees_rotated = 45
    front_face_pts = [top_left_coordinates, (top_left_coordinates[0]+width_cube, top_left_coordinates[1]), (top_left_coordinates[0]+width_cube, top_left_coordinates[1]+width_cube), (top_left_coordinates[0],top_left_coordinates[1]+width_cube) ]
    
    
    # front_face_pts = [top_left_coordinates, (top_left_coordinates[0]+width_cube, top_left_coordinates[1]), (top_left_coordinates[0]+width_cube, top_left_coordinates[1]+width_cube), (top_left_coordinates[0],top_left_coordinates[1]+width_cube) ]
    top_face_pts = [top_left_coordinates,(int(top_left_coordinates[0]+width_cube/2),int(top_left_coordinates[1]-width_cube/2)),(int(top_left_coordinates[0]+width_cube*1.5),int(top_left_coordinates[1]-width_cube/2)),(int(top_left_coordinates[0]+width_cube), top_left_coordinates[1])]
    side_face_pts = [(int(top_left_coordinates[0]+width_cube), top_left_coordinates[1]),(int(top_left_coordinates[0]+width_cube*1.5),int(top_left_coordinates[1]-width_cube/2)),(int(top_left_coordinates[0]+width_cube*1.5),int(top_left_coordinates[1]+width_cube/2)),(top_left_coordinates[0]+width_cube, top_left_coordinates[1]+width_cube)]
    
    # 480 wide by 640 high
    img = img = np.zeros((480, 640, 3), np.uint8)
    cv2.fillPoly(img, np.array([front_face_pts]), ORANGE)
    cv2.fillPoly(img, np.array([top_face_pts]), GREEN)
    cv2.fillPoly(img, np.array([side_face_pts]), BLUE)
    cv2.imshow('window', img)
    if cv2.waitKey(5) == ord("w"):
        top_left_coordinates[1] -=10
    if cv2.waitKey(5) == ord("s"):
        top_left_coordinates[1] +=10
    if cv2.waitKey(5) == ord("a"):
        top_left_coordinates[0] -=10
    if cv2.waitKey(5) == ord("d"):
        top_left_coordinates[0] +=10
cv2.destroyAllWindows()