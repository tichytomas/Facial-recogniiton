import numpy as np
import pickle
import cv2 as cv


#pouziji haar kaskady od volne dostupne od OpenCv, ja si vybral f_f_alt2 
oblicej_cascade = cv.CascadeClassifier(
    r"C:\Users\uzik\python_projects\facial_recognition\cascades\data\haarcascade_frontalface_alt2.xml")

rozpoznavac  = cv.face.LBPHFaceRecognizer_create()
rozpoznavac.read("trenovac.yml")

znacky = {}
with open("labels.pickle","rb") as f:
    znacky = pickle.load(f)
    znacky = {v:k for k,v in znacky.items()} #prehodi poradi cisla a jsmena ve slovniku

kamera = cv.VideoCapture(0)

while True:
    funguje, video = kamera.read()  #inicializace kamery (funguje=True/False)
    gray = cv.cvtColor(video, cv.COLOR_BGR2GRAY)

    #pouziti cv a kaskad na rozliseni obliceje v obraze
    obliceje = oblicej_cascade.detectMultiScale(gray, scaleFactor=1.5,minNeighbors=5,)

    for (x,y,w,h) in obliceje:
        print (x,y,w,h) #vypise souradnice jakehokoli detekovaneho obliceje
        roi = gray[y:y+h,x:x+w] #"region of interest"

        id_, conf = rozpoznavac.predict(roi)
        if conf>=45 and conf<=85:
            print(id_)
            print(znacky)

            font = cv.FONT_HERSHEY_TRIPLEX
            cv.putText(video, znacky[id_], (x,y-3), font, 0.5, (255,255,255), 1, cv.LINE_AA)

        oblicej_img = "muj_ksicht.png"
        cv.imwrite(oblicej_img,roi) #ulozi roi detekovaneho obliceje

        #ramecek kolem obliceje
        konc_sourad_x = x + w 
        konc_sourad_y = y + h
        cv.rectangle(video, (x,y), (konc_sourad_x,konc_sourad_y),(255,255,255),1)
        

    # Spusteni kamery
    cv.imshow('Video',video)

    # Pri zmacknuti "q" se loop prerusi
    if cv.waitKey(20) & 0xFF == ord('q'):
        break


kamera.release()
cv.destroyAllWindows()