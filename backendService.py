from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition
import os
import math
import time

app = Flask(__name__)
socketioApp = SocketIO(app)

# create images array and names
path="faceImages"
images=[]
names=[]
myList = os.listdir(path)

# Look ath the path and append the images and names to the arrays
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    names.append(os.path.splitext(cl)[0]) # get name only

# this function takes the non linear value of the face distance and maps it to a percentage value.
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))


# Function that returns an array of encodings for each image.
def findEncodings(images):
    encodeList =[]
    for img in images:
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

facesKnownEncoded= findEncodings(images)

# video capture
cap = cv2.VideoCapture(0)

def processFrames(compress_rate=0.25):
    counter = 0
    fps = 0
    current_time = time.time()#Save an initial current_time, and update the current_time only after the while true loops twice
    while True:
        success, img = cap.read()
        if counter == 0:
            imgS = cv2.resize(img, (0, 0), None, compress_rate, compress_rate)
            imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
            facesInCurrentFrame = face_recognition.face_locations(imgS)
            encodingsCurrentFrame = face_recognition.face_encodings(imgS, facesInCurrentFrame)
            for encodeFace, faceLocation in zip(encodingsCurrentFrame, facesInCurrentFrame):
                matches = face_recognition.compare_faces(facesKnownEncoded, encodeFace)
                faceDist = face_recognition.face_distance(facesKnownEncoded, encodeFace)
                matchIndex = np.argmin(faceDist)
                if matches[matchIndex]:
                    name = names[matchIndex].upper()
                    matchPercentage = round(face_distance_to_conf(faceDist[matchIndex]) * 100)
                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = int(y1 / compress_rate), int(x2 / compress_rate), int(y2 / compress_rate), int(x1 / compress_rate)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, name + " " + str(matchPercentage) + "%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1,(255, 255, 255), 2)
                else:
                    y1, x2, y2, x1 = faceLocation
                    y1, x2, y2, x1 = int(y1 / compress_rate), int(x2 / compress_rate), int(y2 / compress_rate), int(x1 / compress_rate)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                    cv2.putText(img, "Unknown", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        counter += 1
        if counter == 2:
            fps = 2.0 / (time.time() - current_time) #two frames per seconds
            counter = 0
            current_time = time.time()
        cv2.putText(img, "FPS: {:.2f}".format(fps), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/displayFrames')
def displayFrames():
    return Response(processFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html')

def run():
    socketioApp.run(app)

if __name__ == '__main__':
    socketioApp.run(app)
