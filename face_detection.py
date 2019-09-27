import cv2
import sys
import os

faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

video_capture = cv2.VideoCapture(0)

count=1
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )
    # Draw a rectangle around the faces
    padding=50
    for (x, y, w, h) in faces:
        cv2.rectangle(frame,(x-padding,y-padding),(x+w+padding,y+h+padding),(255,0,0),2)
        face=frame[y-50:y+h+50, x-50:x+w+50] 
        face_resize = cv2.resize(face, (1000, 1000)) 
        cv2.imwrite('a.png', face_resize) 
        #count += 1

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
