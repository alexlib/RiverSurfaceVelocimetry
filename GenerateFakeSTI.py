import cv2
import numpy as np
import os

# width, height = 600, 600
#
# img = np.zeros((width, height), float)
# step = 5
# for i in range(0, height, step):
#     img[i, :] = (256.0 / (i % step + 1)) / 256

img = cv2.imread("sti_imgs/fai=71.6.png")
height, width = img.shape[:2]

rotate_angle = -40
M = cv2.getRotationMatrix2D(((width - 1) / 2.0, (height - 1) / 2.0), rotate_angle, 1)
img_rotate = cv2.warpAffine(img, M, (width, height))
cv2.imshow('img_merge', img_rotate/255)
while True:
    if cv2.waitKey(1000):
        break
cv2.destroyAllWindows()