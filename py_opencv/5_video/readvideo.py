import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# print(cv.VideoCapture.get(0))

while cap.isOpened():
    ret, frame = cap.read()
    print(ret)
    if ret:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        cv.imshow('frame', gray)
        if cv.waitKey(0) & 0xFF == ord('q'):
            break

cap.release()
cv.destroyAllWindows()


