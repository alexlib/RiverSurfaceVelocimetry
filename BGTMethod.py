import cv2
import numpy as np

img = cv2.imread('sti_imgs/fai=11.6.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def calPartialDerivative(I, axis=0, order=4):
    """
    :param I: 待求偏导的灰度图片像素矩阵
    :param axis: 求偏导方向，默认x方向
    :param order: 求偏导方法，分为4阶或5阶，详参考Fujita-2007论文
    :return: 对应偏导矩阵
    """

    def calDerivative(I, index, order):
        """
        :param I: 待求偏导图像行向量
        :param index: 求偏导位置
        :param order: 差分偏导阶次，具体公式见注释
        :return: 对应index位置的差分偏导值
        """
        # df / dx = (f(x + 2) - 8f(x+1) + 8f(x - 1) - f(x - 2)) / 12delta_x
        # df / dx = (f(x + 3) - 9f(x+1) + 45(f(x + 1) - f(x - 1)) - 9f(x - 2) - f(x - 3)) / 60delta_x
        if order == 4:
            return (I[index + 2] - 8 * I[index + 1] + 8 * I[index - 1] - I[index - 2]) / 12
        else:
            return (I[index + 3] - 9 * I[index + 1] + 45 * (I[index + 1] - I[index - 1]) + 9 * I[index - 2] - I[
                index - 3]) / 60

    I = I.T if axis == 1 else I
    width, height = I.shape[:2]
    I_x = np.zeros((width, height), np.float)
    indent = 2 if order == 4 else 3

    # for i in range(indent, width - indent):
    #     I_line = I[i, :]
    #     for j in range(indent, height - indent):
    #         I_x[i, j] = calDerivative(I_line, j, order)

    for i in range(width):
        for j in range(height):
            if j == 0:
                I_x[i, j] = I[i, j+1] - I[i, j]
            elif j == 1:
                I_x[i, j] = (I[i, j+1] - I[i, j-1]) / 2
            elif j == 2:
                    I_x[i, j] = calDerivative(I[i, :], j, 4)
            elif 2 < j < height - 3:
                I_x[i, j] = calDerivative(I[i, :], j, order)
            elif j == height - 3:
                I_x[i, j] = calDerivative(I[i, :], j, 4)
            elif j == height - 2:
                I_x[i, j] = (I[i, j+1] - I[i, j-1]) / 2
            else:
                I_x[i, j] = I[i, j] - I[i, j-1]

    return I_x.T if axis == 1 else I_x


def calJIntegral(I1, I2):
    """
    计算在I定义的区域上I1*I2的积分
    :param I1: 偏导数1，可为I关于x或t的偏导
    :param I2: 偏导数2可为I关于x或t的偏导
    :return: I1*I2在I定义区域上的积分
    """
    I = I1 * I2
    Jxx = 0
    for i in range(I.shape[0]-1):
        for j in range(I.shape[1]-1):
            Jxx += (I[i, j] + I[i, j+1] + I[i+1, j] + I[i+1, j+1]) / 4
    return Jxx


def calTextureAngle(img, order=4):
    """
    cal the texture angle of the input img
    :param img: 输入图像像素矩阵
    :param order: 求导阶次
    :return: 纹理角fai（角度）和图像清晰度C
    """
    img = img.astype(np.int)
    I_x = calPartialDerivative(img, axis=0, order=order)
    I_t = calPartialDerivative(img, axis=1, order=order)
    Jxx = calJIntegral(I_x, I_x)
    Jxt = calJIntegral(I_x, I_t)
    Jtt = calJIntegral(I_t, I_t)
    fai = np.math.atan(2 * Jxt / (Jtt - Jxx)) / 2 / np.math.pi * 180 + 90
    C = np.math.sqrt((Jxx - Jtt) ** 2 + 4 * Jxt ** 2) / (Jxx + Jtt)
    return fai, C


# order为求偏导阶次，I_x, I_t = img(x,t)分别对x,t求偏导矩阵
# order = 4
# img = img.astype(np.int)
# I_x = calPartialDerivative(img, axis=0, order=order)
# I_t = calPartialDerivative(img, axis=1, order=order)
# # 赵浩源版的偏度计算方法
# I_x = cv2.Sobel(img, -1, 1, 0).astype(np.int)
# I_t = cv2.Sobel(img, -1, 0, 1).astype(np.int)
#
# # Jxx = I_x*I_x在img上的二重积分，Jxt, Jtt类推
# Jxx = calJIntegral(I_x, I_x)
# Jxt = calJIntegral(I_x, I_t)
# Jtt = calJIntegral(I_t, I_t)
# # fai 纹理角
# fai = np.math.atan(2 * Jxt / (Jtt - Jxx)) / 2 / np.math.pi * 180 + 90
# C = np.math.sqrt((Jxx - Jtt) ** 2 + 4 * Jxt ** 2) / (Jxx + Jtt)
fai, C = calTextureAngle(img, 4)
print("fai = %.2f°, tan(fai) = %.2f, C = %.2f" % (fai, np.math.tan(fai / 180 * np.math.pi), C))

