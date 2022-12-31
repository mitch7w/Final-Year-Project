from pydoc import plain
from PIL import Image, ImageDraw
import PIL  # for .png loading and conversion to ndarray
import numpy as np

newImage = Image.open("prototypes/Shape Generation/room.jpg").convert('RGB')
# newImage.show()
newImageArray1 = np.array(newImage) # 2128 height x 4608 width by 3 rgb pixels at each point
# draw orange (255, 215, 0) square 
orangePixel = [255,125,0]
# for row in range(1500,2000):
#     for col in range(2500,3000):
#         newImageArray1[row][col] = orangePixel
# orangeSquareImage = Image.fromarray(newImageArray1.astype(np.uint8))
# orangeSquareImage.show()
# draw green cube
newImageArray2 = np.array(newImage) # 2128 height x 4608 width by 3 rgb pixels at each point
greenPixel = [0,240,0]
# left and right lines
for row in range(1500,2000):
    newImageArray2[row][2500] = greenPixel
    newImageArray2[row][3000] = greenPixel
# top and bottom lines
for col in range(2500,3000):
    newImageArray2[1500][col] = greenPixel
    newImageArray2[2000][col] = greenPixel
# top 3d projection
for col in range(2750,3250):
    newImageArray2[1250][col] = greenPixel
#left diagonal
rowCounter = 1500
for col in range(2500,2750):
        newImageArray2[rowCounter][col] = greenPixel
        rowCounter -=1
#right diagonal
rowCounter = 1500
for col in range(3000,3250):
        newImageArray2[rowCounter][col] = greenPixel
        rowCounter -=1
# bottom diagonal
rowCounter = 2000
for col in range(3000,3250):
        newImageArray2[rowCounter][col] = greenPixel
        rowCounter -=1
#rightmost vertical line
for row in range(1250,1750):
    newImageArray2[row][3250] = greenPixel
greenCubeImage = Image.fromarray(newImageArray2.astype(np.uint8))
greenCubeImage.show()