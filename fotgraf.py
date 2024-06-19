import numpy as np
import cv2 as cv
import os

jmeno = f"\{'tomas'}"
os.mkdir(r"C:\Users\uzik\python_projects\facial_recognition\obliceje"+(jmeno))

oblicej_cascade = cv.CascadeClassifier(
    r"C:\Users\uzik\python_projects\facial_recognition\cascades\data\haarcascade_frontalface_alt2.xml")

kamera = cv.VideoCapture(0)

photo_count = 1

while True:
    funguje, video = kamera.read()  #inicializace kamery (funguje=True/False)
    gray = cv.cvtColor(video, cv.COLOR_BGR2GRAY)

    obliceje = oblicej_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5,)

    for (x,y,w,h) in obliceje:
        roi = gray[y:y+h,x:x+w] #"region of interest"
        foto_roi = video[y-50:y+h+50,x-50:x+w+50]

        konc_sourad_x = x + w 
        konc_sourad_y = y + h
        cv.rectangle(video, (x-50,y-50), (konc_sourad_x+50,konc_sourad_y+50),(255,255,255),1)

    cv.imshow('Video',video)


    if cv.waitKey(20) & 0xFF == ord('q'):
        break

    if cv.waitKey(20) & 0xFF == ord('p'):
        fotka = f"facial_recognition\obliceje{jmeno}\{photo_count}.jpg"
        cv.imwrite(fotka,video)
        photo_count +=1

kamera.release()
cv.destroyAllWindows()
