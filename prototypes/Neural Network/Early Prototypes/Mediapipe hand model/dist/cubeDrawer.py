import time
import csv
import cv2

def drawCube(x,y,z):
   global image
   # draw cube according to pink co-ords
   x_cord = int((x*1000)+100)
   y_cord = int((y*1000)+100)
   start_point = (x_cord, y_cord) # start co-ord top left of rectangle
   end_point = (x_cord+50, y_cord+50) # end co-ord bottom right of rectangle
#    x is -0.2 to 1
# -0.2 at top to 1.2 at bottom
   image = cv2.rectangle(image, start_point, end_point, color=(255, 0, 0), thickness=2)
   # Displaying the image 
   cv2.imshow("Cube Image", image)
   cv2.waitKey(100)

def followFile(thefile):
    while True:
        line = thefile.readline()
        if line:
            reader = csv.reader([line], delimiter=',')
            for row in reader:
                print(row[1])
                drawCube(float(row[0]),float(row[1]),float(row[2]))
        else:
            time.sleep(0.1)
            continue

image = cv2.imread("beach.jpeg")
textFile = open("handData.txt", "r")
followFile(textFile)