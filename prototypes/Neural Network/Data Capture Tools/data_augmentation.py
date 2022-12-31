import cv2
import numpy as np
import scipy
from scipy.ndimage import rotate

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img

imageName = "/Users/mitch/Documents/University/Project/kinect_side_open_fist_massive/gesture1.jpg"
image = cv2.imread(imageName,1) 
rotatedImage = rotate_img(image,45)
while True:
    cv2.imshow("Image", image)
    cv2.imshow("Rotated image", rotatedImage)
    cv2.waitKey(500)

# # loop through whole dataset of images and create vertically flipped versions
# for i in range(1,801):
#     imageName = "/Users/mitch/Documents/University/Project/kinect_side_open_fist_massive/gesture" + str(i) + ".jpg"
#     image = cv2.imread(imageName,1) 
#     flipVertical = cv2.flip(image, 1)
#     flipHorizontal = cv2.flip(image, 0)
#     flipBoth = cv2.flip(image, -1)
#     image_filepath = "/Users/mitch/Documents/University/Project/kinect_side_open_fist_massive/gesture" + str(i+800) + ".jpg"
#     image_filepath1 = "/Users/mitch/Documents/University/Project/kinect_side_open_fist_massive/gesture" + str(i+1600) + ".jpg"
#     image_filepath2 = "/Users/mitch/Documents/University/Project/kinect_side_open_fist_massive/gesture" + str(i+2400) + ".jpg"
#     cv2.imwrite(image_filepath, flipVertical)
#     cv2.imwrite(image_filepath1, flipHorizontal)
#     cv2.imwrite(image_filepath2, flipBoth)
#     # loop through whole dataset of images and create vertically flipped versions

    