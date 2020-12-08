import cv2
import numpy as np


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
        # df / dx = (f(x + 3) - 9f(x+2) + 45(f(x + 1) - f(x - 1)) + 9f(x - 2) - f(x - 3)) / 60delta_x
        if order == 4:
            return (I[index + 2] - 8 * I[index + 1] + 8 * I[index - 1] - I[index - 2]) / -12
        else:
            b = 3
            c = -27 + 8 * b
            return ((I[index + 3] - I[index - 3]) - b * (I[index + 2] - I[index - 2])
                    + c * (I[index + 1] - I[index - 1])) / (6 - 4 * b + 2 * c)

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
                I_x[i, j] = I[i, j + 1] - I[i, j]
            elif j == 1:
                I_x[i, j] = (I[i, j + 1] - I[i, j - 1]) / 2
            elif j == 2:
                I_x[i, j] = calDerivative(I[i, :], j, 4)
            elif 2 < j < height - 3:
                I_x[i, j] = calDerivative(I[i, :], j, order)
            elif j == height - 3:
                I_x[i, j] = calDerivative(I[i, :], j, 4)
            elif j == height - 2:
                I_x[i, j] = (I[i, j + 1] - I[i, j - 1]) / 2
            else:
                I_x[i, j] = I[i, j] - I[i, j - 1]

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
    for i in range(I.shape[0] - 1):
        for j in range(I.shape[1] - 1):
            Jxx += (I[i, j] + I[i, j + 1] + I[i + 1, j] + I[i + 1, j + 1]) / 4
    return Jxx


def calTextureAngle(img, order=4, stride=None):
    """
    cal the texture angle of the input img
    :param img: 输入图像像素矩阵
    :param order: 求导阶次
    :param stride: 切分图片步长，分为x和t两个方向，每个方向值必须大于10且小于宽或高；仅x时,t用x的值替代
    :return: 纹理角fai（角度）和图像清晰度C
    """
    img = img.astype(np.int)
    width, height = img.shape[:2]
    I_x = calPartialDerivative(img, axis=0, order=order)
    I_t = calPartialDerivative(img, axis=1, order=order)

    if stride is None:
        stride_x, stride_y = width, height
    elif len(stride) == 1:
        stride_x, stride_y = max(min(stride[0], width), 10), max(min(stride[0], height), 10)
    else:
        stride_x, stride_y = max(min(stride[0], width), 10), max(min(stride[1], height), 10)

    fai_list, C_list = [], []
    for i in range(width // stride_x):
        for j in range(height // stride_y):
            if i == width // stride_x - 1 and j == height // stride_y - 1:
                I_x_part = I_x[i * stride_x::, j * stride_y::]
                I_t_part = I_t[i * stride_x::, j * stride_y::]
            elif i == width // stride_x - 1 and j != height // stride_y - 1:
                I_x_part = I_x[i * stride_x::, j * stride_y: (j + 1) * stride_y]
                I_t_part = I_t[i * stride_x::, j * stride_y: (j + 1) * stride_y]
            elif i != width // stride_x - 1 and j == height // stride_y - 1:
                I_x_part = I_x[i * stride_x: (i + 1) * stride_x, j * stride_y::]
                I_t_part = I_t[i * stride_x: (i + 1) * stride_x, j * stride_y::]
            else:
                I_x_part = I_x[i * stride_x: (i + 1) * stride_x, j * stride_y: (j + 1) * stride_y]
                I_t_part = I_t[i * stride_x: (i + 1) * stride_x, j * stride_y: (j + 1) * stride_y]

            Jxx_part = calJIntegral(I_x_part, I_x_part)
            Jxt_part = calJIntegral(I_x_part, I_t_part)
            Jtt_part = calJIntegral(I_t_part, I_t_part)

            fai_list.append((np.math.atan(2 * Jxt_part / (Jtt_part - Jxx_part)) / 2 / np.math.pi * 180 + 90) % 90)
            C_list.append(np.math.sqrt((Jxx_part - Jtt_part) ** 2 + 4 * Jxt_part ** 2) / (Jxx_part + Jtt_part))

    fai_list = np.array(fai_list).reshape((1, -1))
    C_list = np.array(C_list).reshape((1, -1))
    fai = (fai_list * C_list).sum() / C_list.sum()
    return fai, sum(C_list) / len(C_list)


if __name__ == '__main__':
    real_fai = 11.6
    img = cv2.imread('sti_imgs/fai=%s.png' % str(real_fai))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fai, _ = calTextureAngle(img, 5)
    error_v = abs(1 - np.math.tan(fai / 180 * np.math.pi) / np.math.tan(real_fai / 180 * np.math.pi)) * 100
    delta_fai = abs(fai - real_fai)
    error_fai = abs(1 - fai / real_fai) * 100
    print("fai = %.2f°, tan(fai) = %.2f" % (fai, np.math.tan(fai / 180 * np.math.pi)))
    print("delta_fai = %.2f°, error_fai = %.2f%%, error_v = %.2f%%" % (delta_fai, error_fai, error_v))

