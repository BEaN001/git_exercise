import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# print(cv.VideoCapture.get(0))

width = 224
height = 224
dim = (width, height)

while cap.isOpened():
    ret, frame = cap.read()
    print(ret)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    resize_gray = cv.resize(gray, dim, interpolation=cv.INTER_AREA)

    cv.imshow('frame', resize_gray)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()


