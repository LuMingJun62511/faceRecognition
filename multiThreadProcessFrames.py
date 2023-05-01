import queue
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition
import os
import math
import time
import threading

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


# taken from https://github.com/ageitgey/face_recognition/wiki/Calculating-Accuracy-as-a-Percentage

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


# Function that returns a array of encodings for each image.
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

def readFrames():
    while True:
        success, img = cap.read()
        if success:
            frame_queue.put(img)

def inferFaces(compress_rate=0.25):
    while True:
        img = frame_queue.get()
        current_time = time.time()

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
                detected_faces.append((name, matchPercentage, x1, y1, x2, y2))
            else:
                y1, x2, y2, x1 = faceLocation
                y1, x2, y2, x1 = int(y1 / compress_rate), int(x2 / compress_rate), int(y2 / compress_rate), int(x1 / compress_rate)
                detected_faces.append(("Unknown", 0, x1, y1, x2, y2))

        fps_text = "FPS: {:.2f}".format(1.0 / (time.time() - current_time))
        fps_queue.put(fps_text)

def returnFrames():
    while True:
        img = frame_queue.get()
        faces = detected_faces.copy()
        detected_faces.clear()
        fps_text = fps_queue.get()

        for face in faces:
            name, matchPercentage, x1, y1, x2, y2 = face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name + " " + str(matchPercentage) + "%", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(img, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        ret, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        yield b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n'

frame_queue = queue.Queue()
fps_queue = queue.Queue()
detected_faces = []

# Start threads
threading.Thread(target=readFrames, daemon=True).start()
threading.Thread(target=inferFaces, daemon=True).start()
threading.Thread(target=returnFrames, daemon=True).start()


@app.route('/displayFrames')
def displayFrames():
    return Response(returnFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    # return render_template('index.html', client_count=client_count)
    return render_template('index.html')

def run():
    socketioApp.run(app)

if __name__ == '__main__':
    socketioApp.run(app)


