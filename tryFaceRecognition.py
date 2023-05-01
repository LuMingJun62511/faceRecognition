import cv2
import face_recognition

asStandard= face_recognition.load_image_file("testImages/AlPacino56.jpg")
asStandard = cv2.cvtColor(asStandard, cv2.COLOR_BGR2RGB)

toRecog= face_recognition.load_image_file("testImages/AlPacino35.jpg")
toRecog =cv2.cvtColor(toRecog, cv2.COLOR_BGR2RGB)


faceLocation = face_recognition.face_locations(asStandard)[0]
asStandardEncoded=face_recognition.face_encodings(asStandard)[0]
cv2.rectangle(asStandard, (faceLocation[3], faceLocation[0]), (faceLocation[1], faceLocation[2]), (0, 255, 0), 2)


faceLocationTest = face_recognition.face_locations(toRecog)[0]
toRecogEncoded=face_recognition.face_encodings(toRecog)[0]
cv2.rectangle(toRecog, (faceLocationTest[3], faceLocationTest[0]), (faceLocationTest[1], faceLocationTest[2]), (0, 255, 0), 2)


results= face_recognition.compare_faces([asStandardEncoded], toRecogEncoded) # see if its the same person

faceDist= face_recognition.face_distance([asStandardEncoded], toRecogEncoded)
print(results,faceDist)

cv2.putText(toRecog, f'{results} {round(faceDist[0], 2)}', (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

cv2.imshow('Al Pacino img as the standard ', asStandard)
cv2.imshow('Al Pacino img to be recognized', toRecog)

cv2.waitKey(0)






