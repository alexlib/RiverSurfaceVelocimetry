import cv2
import numpy as np
import matplotlib.pyplot as plt


def RecTangle2Polar(img, shape):
    """
    将输入的灰度图片矩阵由直角坐标系变为极坐标系表示，
    其中原坐标原定默认为图片中心，
    非整数点用双线性插值求解
        :param img: 输入灰度图片矩阵
        :param shape: 输出图片的尺寸，其中长不小于原图片最短边的一半
    :return: 极坐标表示的图片矩阵
    """

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    width, height = img.shape
    length = min(width, height)
    xCenter, yCenter = width // 2, height // 2

    radius_nums = max(length // 2, shape[0])
    theta_nums = shape[1]
    # theta_nums = max(360, shape[1])
    dRadius, dTheta = length / radius_nums, 360 / theta_nums
    dst = np.zeros([radius_nums, theta_nums], np.float)

    for i in range(radius_nums):
        for j in range(theta_nums):
            # print(i, j)
            if i == 0:
                dst[i][j] = img[xCenter][yCenter]
            else:
                delta_x = np.sqrt(np.power(i * dRadius, 2) / (1 + np.power(np.tan(j * dTheta / 180 * np.pi), 2)))
                delta_y = np.sqrt((np.power(i * dRadius, 2) * np.power(np.tan(j * dTheta / 180 * np.pi), 2)) /
                                  (1 + np.power(np.tan(j * dTheta / 180 * np.pi), 2)))

                if 0 <= j < theta_nums / 4 or theta_nums * 3 / 4 < j < theta_nums:
                    x = delta_x + xCenter
                else:
                    x = delta_x - xCenter
                if 0 <= j < theta_nums / 2:
                    y = delta_y + yCenter
                else:
                    y = delta_y - yCenter

                xCeil, xFloor = int(min(np.ceil(x), width - 1)), int(min(np.floor(x), width - 1))
                yCeil, yFloor = int(min(np.ceil(y), height - 1)), int(min(np.floor(y), height - 1))

                if yFloor == yCeil and xFloor == xCeil:
                    out = img[xCeil][yCeil]
                elif yFloor == yCeil:
                    out = (x - xFloor) * img[xFloor][yCeil] + (xCeil - x) * img[xFloor][yFloor]
                elif xFloor == xCeil:
                    out = (y - yFloor) * img[xCeil][yFloor] + (yCeil - y) * img[xFloor][yFloor]
                else:
                    inter1 = (x - xFloor) * img[xFloor][yCeil] + (xCeil - x) * img[xFloor][yFloor]
                    inter2 = (x - xFloor) * img[xCeil][yCeil] + (xCeil - x) * img[xCeil][yFloor]
                    out = (y - yFloor) * inter2 + (yCeil - y) * inter1

                dst[i][j] = out

    return dst.T


# img = cv2.imread('1.png')
# img_polar = RecTangle2Polar(img, [max(img.shape), 360])
# plt.imshow(img_polar, 'gray')
# plt.show()
