import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

id = 3
# 图像预处理，长宽需要一致，最好都是偶数
fp_read, fp_write = '12.4 stiv images/%d_%d_599.jpg' % (id, 90 + id), '%d_%d_599_filtered.jpg' % (id, 90 + id)
img = plt.imread(fp_read)
img = img[:, 100:400]

height = np.array(img).shape[0]
width = np.array(img).shape[1]

if len(np.array(img).shape) == 3:
    img = 0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
# img = cv2.equalizeHist(img)


# cv2.imshow('原始图像', img)
plt.subplot(121)
plt.imshow(img, 'gray')
plt.title('original')

img_fft = np.fft.fft2(img)
img_shiftcenter = np.fft.fftshift(img_fft)
img_logshiftcenter = np.log(1 + np.abs(img_shiftcenter))
# cv2.imshow('傅里叶图谱，原点移动到中心', np.abs(img_fft))
plt.subplot(122)
plt.imshow(img_logshiftcenter, 'gray')
plt.title('fourier')
cv2.imwrite(fp_write.replace('filtered', 'fourier'), img_logshiftcenter)
plt.show()


# 获取一条线上所有像素点
def createLineIterator(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(np.int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(np.int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(np.uint), itbuffer[:, 0].astype(np.uint)]

    return itbuffer


# 计算angel角度的线上像素均值
def line_integral(angel, img):
    angel = angel / 180 * math.pi
    x = width / 2 * math.cos(angel)
    y = height / 2 * math.sin(angel)
    x1 = math.floor(width / 2 + x)
    x2 = math.ceil(width / 2 - x)
    y1 = math.ceil(height / 2 - y)
    y2 = math.floor(height / 2 + y)

    itbuffer = createLineIterator(np.array([x1, y1]), np.array([x2, y2]), img)
    itbuffer_len = len(itbuffer)
    total = sum(itbuffer[:, 2]) / itbuffer_len
    return total


# 求theta，只能识别10到80度范围
array = np.zeros([90])
for angel in range(10, 85):
    array[angel] = line_integral(angel, img_logshiftcenter)
theta = np.argmax(array)
print(max(array))
print(theta)
# theta = 71
plt.plot(range(10, 85), array[10:85])
plt.show()

# 求解mask
mask = np.zeros(img_shiftcenter.shape, dtype=np.uint8)
beta = 20
min_angel = theta - beta / 2
max_angel = theta + beta / 2
# mask[int(height/2)][int(width/2)]=1
for i in range(width):
    for j in range(height):
        if i != int(width / 2) and j != int(height / 2):
            angel_tan = (height / 2 - j) / (i - width / 2)
            angel = math.atan(angel_tan) / math.pi * 180
            distance = (i - width / 2) ** 2 + (j - height / 2) ** 2
            if distance <= width ** 2 / 8 and min_angel < angel < max_angel:
                mask[j][i] = 1

img_2 = cv2.equalizeHist(img)
plt.subplot(121)
plt.imshow(img_2, 'gray')
plt.title('equalizeHist')
cv2.imwrite(fp_write.replace('filtered', 'equalizeHisted'), img_2)

img_fft_2 = np.fft.fft2(img_2)
img_shiftcenter_2 = np.fft.fftshift(img_fft_2)
# img_logshiftcenter_2 = np.log(1+np.abs(img_shiftcenter_2))

new_fft2_2 = img_shiftcenter_2 * mask
new_fft1_2 = np.fft.ifftshift(new_fft2_2)
new_img_2 = np.fft.ifft2(new_fft1_2)
new_img_2 = np.abs(new_img_2)
plt.subplot(122)
plt.imshow(new_img_2, 'gray')
plt.title('result2')
plt.show()
cv2.imwrite(fp_write.replace('filtered', 'result'), new_img_2)

# 画图显示结果
new_fft2 = img_shiftcenter * mask
new_fft1 = np.fft.ifftshift(new_fft2)
new_img = np.fft.ifft2(new_fft1)
new_img = np.abs(new_img)

plt.subplot(121)
plt.imshow(np.log(1 + np.abs(new_fft2)), 'gray')
plt.title('mask')

plt.subplot(122)
plt.imshow(new_img, 'gray')
plt.title('result')
cv2.imwrite(fp_write.replace('filtered', 'result_new'), new_img)
plt.show()
