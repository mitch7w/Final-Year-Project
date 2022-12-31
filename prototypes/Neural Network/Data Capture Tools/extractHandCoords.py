import mediapipe
import cv2
import numpy as np


# on spacebar get handCoords of current gesture

handCounter = 0

def extractCoords(image , handCoordinateArray):
    global handCounter
    # handCoordinateArray[0].x = x coord of wrist
    x=[]
    y=[]
    z=[]
    for i in range(21):
        x.append(handCoordinateArray[i].x)
        y.append(handCoordinateArray[i].y)
        z.append(handCoordinateArray[i].z)
    handCoords = [x,y,z]
    image_filepath = "handcoordstrainingdata/gesture" + str(handCounter) + ".jpg"
    cv2.imwrite(image_filepath, image)
    coord_filepath = "handcoordstrainingdata/gesture" + str(handCounter)
    np.save(coord_filepath, handCoords)
    handCounter += 1

mp_drawing = mediapipe.solutions.drawing_utils
mp_drawing_styles = mediapipe.solutions.drawing_styles
mp_hands = mediapipe.solutions.hands

cap = cv2.VideoCapture(0)
pinkyCoords = {}
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    resized = cv2.resize(image, (160,120), interpolation = cv2.INTER_AREA)
    handCoords=[]
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        handCoords = hand_landmarks.landmark
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(resized, 1))
    if cv2.waitKey(5) == ord(" "):
      extractCoords(resized,handCoords)
cap.release()
