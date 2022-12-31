from flask import Flask, request, jsonify
from flask_cors import CORS

#Set up Flask
app = Flask(__name__)

#Set up Flask to bypass CORS
cors = CORS(app)

#Create the receiver API POST endpoint:
@app.route("/receiver", methods=["POST"])

def postME():
   data = request.get_json()
   data = jsonify(data)
   return data

if __name__ == "__main__": 
   app.run(debug=True)