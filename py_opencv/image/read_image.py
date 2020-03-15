import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

print("------read, show image with opencv BGR-------")
# img = cv.imread("./opencvv.png", 0)
# cv.imshow('image', img)
# cv.waitKey(0)
# k = cv.waitKey(0)
# if k == 27:  # wait for ESC key to exit
#     cv.destroyAllWindows()
# elif k == ord('s'):  # wait for 's' key to save and exit
#     cv.imwrite('./messigray.png', img)
#     cv.destroyAllWindows()

print("------show image with pyplot RGB---------")
img = cv.imread("./opencvv.png")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB) # opencv BGR to RGB
plt.imshow(img)
plt.show()

