import cv2
import numpy as np

cap = cv2.VideoCapture('d:/downloads/VID_20201204_160411.mp4')
xmin, xmax, y = 678, 818, 652
img = np.zeros((xmax-xmin, xmax-xmin, 3), np.float32)
frame_num = xmax - xmin
ret, frame = cap.read()


# def ON_EVENT_LBUTTONDOWN(event, x, y, flags, param):
#     print(x, y)
#
#
# cv2.namedWindow('img')
# cv2.setMouseCallback('img', ON_EVENT_LBUTTONDOWN)
# cv2.imshow('img', frame)
# cv2.waitKey(0)

while frame_num > 0:
    ret, frame = cap.read()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if ret:
        img[xmax-xmin-frame_num, :, :] = frame[y, xmin:xmax, :]
    frame_num -= 1

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('sti img', img/255)
while True:
    if cv2.waitKey(0):
        break
cv2.destroyAllWindows()