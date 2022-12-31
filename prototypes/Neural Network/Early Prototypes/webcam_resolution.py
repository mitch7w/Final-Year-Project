import cv2
# 1280x720 or 640x480
cv2.namedWindow("Webcam Input")
cam = cv2.VideoCapture(0)
# default is 720x1280
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # set new dimensionns to cam object (not cap)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(width, height)

if(cam).isOpened(): # try to get the first frame
    rval, frame = cam.read()
else:
    rval = False
imageCounter = 0
while rval:
    resized = cv2.resize(frame, (160,120), interpolation = cv2.INTER_AREA)
    cv2.imshow("Webcam Input", resized)
    rval, frame = cam.read()
    key = cv2.waitKey(20)
    if key == ord(' '): # on space save image
        new_filepath = 'hand_pics/hand' + str(imageCounter) + '.jpg'
        cv2.imwrite(new_filepath, resized)
        imageCounter+=1

cam.release()
cv2.destroyWindow("preview")