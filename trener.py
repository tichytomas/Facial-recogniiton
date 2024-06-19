import pickle 
import numpy as np
import os
from PIL import Image
import cv2 as cv


zakl_adresa = os.path.dirname(os.path.abspath(__file__)) # adresa "obliceje_train.py"
obr_adresa = os.path.join(zakl_adresa, "obliceje") # slozka "obliceje"

oblicej_cascade = cv.CascadeClassifier(r"C:\Users\uzik\python_projects\facial_recognition\cascades\data\haarcascade_frontalface_alt2.xml")

rozpoznavac  = cv.face.LBPHFaceRecognizer_create()

aktualni_id = 0
cisla_znacek = {}

trenink = []
znacky = []

# oznacim fotky na uceni a ziskam jejich adresy 
for root, adresy, soubory in os.walk(obr_adresa):
    for soubor in soubory:
        if soubor.endswith("jpg"):
            cesta = os.path.join(root, soubor) #adresy fotek
            znacka = os.path.basename(root).replace(" ", "-") #oznacovani

            if not znacka in cisla_znacek:
            
                cisla_znacek[znacka] = aktualni_id
                aktualni_id +=1
            
            id_ = cisla_znacek[znacka]

        #prevedeni na NUMPY array
        pil_obr = Image.open(cesta).convert("L") # .convert("L") to udela cernobily
        obr_array = np.array(pil_obr, "uint8")
        obliceje = oblicej_cascade.detectMultiScale(obr_array, scaleFactor=1.5,minNeighbors=5,)

        for (x,y,w,h) in obliceje:
            roi = obr_array[y:y+h,x:x+w]
            trenink.append(roi)
            znacky.append(id_)

with open("labels.pickle","wb") as f:
    pickle.dump(cisla_znacek,f)

rozpoznavac.train(trenink, np.array(znacky))
rozpoznavac.save("trenovac.yml")

print("Obličej byl úspěšně zařazen do databáze :) ")