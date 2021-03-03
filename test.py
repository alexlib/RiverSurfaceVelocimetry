import cv2
from skimage.transform import radon
import numpy as np


fai = 31.6
image_path = 'sti_imgs/fai=%s.png' % (str(fai))
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

theta = np.linspace(-90., 90., max(img.shape), endpoint=False)
sinogram = radon(img, theta=theta, circle=True)
sinogram = cv2.normalize(sinogram, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
cv2.imshow('radon', sinogram)
cv2.waitKey(0)




