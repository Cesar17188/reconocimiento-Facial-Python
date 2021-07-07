# Librerias de python
import numpy as np
import cv2

# Clasificador para reconocimientode ojos
# (se puede cambiar por otros reconocimientos)
faceClassif = cv2.CascadeClassifier('haarcascade_eye.xml')

# imagen para reconocer
image = cv2.imread('ojos_mascarilla.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# escalas de la imagen (ajustar a lo que se reconoce)
faces = faceClassif.detectMultiScale(gray,
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(30, 30),
                                     maxSize=(200, 200))

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
