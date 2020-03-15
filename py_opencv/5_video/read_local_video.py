import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)

# cap = cv.VideoCapture('./vtest.avi')

# print(cv.VideoCapture.get(0))

width = 64
height = 128
dim = (width, height)

# Define the codec and create VideoWriter object
fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('./output.avi', fourcc, 20.0, dim)

while cap.isOpened():
    ret, frame = cap.read()
    print(ret, frame.shape)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    resize_gray = cv.resize(gray, dim, interpolation=cv.INTER_AREA)

    cv.imshow('frame', resize_gray)

    resize_ori = cv.resize(frame, dim, interpolation=cv.INTER_AREA)
    out.write(resize_ori)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()


