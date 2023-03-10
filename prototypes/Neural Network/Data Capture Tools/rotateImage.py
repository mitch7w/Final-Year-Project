import cv2
import numpy as np
image = cv2.imread("person.PNG")

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

rotated = rotate_image(image,100)

while True:
    cv2.imshow("Image", image)
    cv2.imshow("Image Rotated", rotated)
    cv2.waitKey(1)