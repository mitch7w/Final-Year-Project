from PIL import Image
import numpy as np
from scipy.signal import convolve2d
from skimage.transform import rescale
import matplotlib.pyplot as plt
from skimage.io import imread, imshow
from skimage.color import rgb2gray

# import image

currentFrameArray = []
with Image.open("person.png") as im:
    currentFrameArray = np.array(im) # 720x1280x4 height x width x rgba

# remove alpha values

newFrameArray = []
for i in range(720):
    newRow = []
    for j in range(1280):
        newRow.append(currentFrameArray[i][j][:-1])
    newRow.append([255,255,255]) # append a white pixel so whole image can be used with 3x3 kernels
    newFrameArray.append(newRow)
newFrameArray = np.array(newFrameArray)

# convolve each color layer
def rgb_convolve2d(image, kernel):
    red = convolve2d(image[:,:,0], kernel, 'valid')
    green = convolve2d(image[:,:,1], kernel, 'valid')
    blue = convolve2d(image[:,:,2], kernel, 'valid')
    return np.stack([red, green, blue], axis=2)

kernel = np.array([[0, 0, 0],
                     [0, 1, 0],
                     [0, 0, 0]])

convolution_output = newFrameArray

r_scaled = rescale(convolution_output[:,:,0], 0.10)
g_scaled = rescale(convolution_output[:,:,1], 0.10)
b_scaled = rescale(convolution_output[:,:,2], 0.10)
image_scaled = np.stack([r_scaled, g_scaled, b_scaled], axis=2)

# apply kernel
conv_im1 = rgb_convolve2d(convolution_output, kernel)
fig, ax = plt.subplots(1,2, figsize=(12,5))
ax[0].imshow(kernel, cmap='gray')
ax[1].imshow(abs(conv_im1), cmap='gray')
plt.show()

