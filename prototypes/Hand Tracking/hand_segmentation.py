import cv2
from PIL import Image
import numpy as np

def redThresholding():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Hand Segmentation Demo")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == ord('x'):
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "prototypes/Hand Segmentation/opencv_frame_{}.png".format(img_counter)
            currentFrame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            currentFrameArray = np.array(currentFrame)
            # build a red thresholder
            numCols = currentFrameArray.shape[0]
            numRows = currentFrameArray.shape[1]
            print("monkeys: ", currentFrameArray.shape)
            for row in range(numRows):
                for col in range(numCols):
                    modifyPixel = True
                    if(currentFrameArray[col][row][0] > 100):
                        if(currentFrameArray[col][row][1] < 50):
                            if(currentFrameArray[col][row][2] < 50):
                                modifyPixel = False
                    if(modifyPixel): # make all non-red pixels white
                        currentFrameArray[col][row][0] = 255
                        currentFrameArray[col][row][1] = 255
                        currentFrameArray[col][row][2] = 255
            currentFrame.show()
            modifiedFrame = Image.fromarray(currentFrameArray)
            modifiedFrame.show()
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

def greyscaleThresholding():
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Hand Segmentation Demo")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("test", frame)

        k = cv2.waitKey(1)
        if k%256 == ord('x'):
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "prototypes/Hand Segmentation/opencv_frame_{}.png".format(img_counter)
            currentFrame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            currentFrameArray = np.array(currentFrame)
            # # build a greyscale binary thresholder
            numCols = currentFrameArray.shape[0]
            numRows = currentFrameArray.shape[1]
            for row in range(numRows):
                for col in range(numCols):
                    if(currentFrameArray[col][row] > 120):
                        currentFrameArray[col][row] = 0
                    else:
                        currentFrameArray[col][row] = 255
            modifiedFrame = Image.fromarray(currentFrameArray)
            modifiedFrame.show()
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()

# greyscaleThresholding()
redThresholding()