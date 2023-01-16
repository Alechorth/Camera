import numpy as np
import cv2
import sys

# Feuilles de descriptions pour le visages et les yeux
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while 1:
    # Image à analyser
    img, frame = cap.read()

    # Convertit l'image en gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte les visages
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(20, 20)
    )

    # Dessine un rectangle bleu autour des visages
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Cherche des yeux dans chaque visage, même principe
        eye = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=15,
            minSize=(25, 25),
        )

        for (xx, yy, ww, hh) in eye:
            cv2.rectangle(roi_color, (xx, yy),
                        (xx + ww, yy + hh), (0, 255, 0), 2)

    # Affiche l'image
    cv2.imshow('image', frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
