from cgitb import text
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import cv2
import matplotlib.pyplot as plt

x_cord = 5
y_cord = 5

# erase file at start
textFile = open("handData.txt", "w").close()


def writePinkyData(landmark):
   outputString = str(landmark['x'])
   outputString += ","
   outputString += str(landmark['y'])
   outputString += ","
   outputString += str(landmark['z'])
   outputString += "\n"
   # append new hand data to file
   textFile = open("handData.txt", "a")
   textFile.write(outputString)
   textFile.close()

#Set up Flask
app = Flask(__name__)

#Set up Flask to bypass CORS
cors = CORS(app)

# #Create the receiver API POST endpoint:
@app.route("/receiver", methods=["POST"])

def postME():
   global x_cord , y_cord
   data = request.get_json()
   # data is list of objects. Each object has x,y,z co-ords. data[20] is pinky.
   writePinkyData(data[20])
   data = jsonify(data)
   return data

if __name__ == "__main__": 
   app.run(debug=True)