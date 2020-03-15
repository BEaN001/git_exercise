import numpy as np
import cv2 as cv

img = cv.imread("./opencvv.png", 0)
cv.imshow('image', img)
cv.waitKey(0)
k = cv.waitKey(0)
if k == 27:  # wait for ESC key to exit
    cv.destroyAllWindows()
elif k == ord('s'):  # wait for 's' key to save and exit
    cv.imwrite('./messigray.png', img)
    cv.destroyAllWindows()


