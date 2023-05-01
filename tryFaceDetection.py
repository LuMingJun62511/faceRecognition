import cv2
import face_recognition

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()

    imgS= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    facesInCurrentFrame = face_recognition.face_locations(imgS) #multiple faces
    encodingsCurrentFrame=face_recognition.face_encodings(imgS,facesInCurrentFrame)

    for encodeFace,faceLocation in zip(encodingsCurrentFrame,facesInCurrentFrame):
        y1, x2, y2, x1 = faceLocation
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Webcam",img)
    cv2.waitKey(1)
