from STAMethod import *
from BGTMethod import *
import glob
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt


# 1800 400 100

def generate_sti(height_start, height_end, height_interp, width_start, width_end, frames=90, frame_start=0,
                 rotate_angle=-45):
    """
    生成STI图像，宽为width_end - width_start，高为frames
        :param height_interp:
        :param height_end:
        :param height_start:
        :param rotate_angle:
        :param height: 原图片取STI像素的高度
        :param width_start: 原图片取STI像素的开始宽度
        :param width_end: 原图片取STI像素的结束宽度
        :param frames: STI图像是多少帧差构成的
        :param frame_start: 从第几帧开始取STI图像
    """
    imgs = []
    # for k in range((height_end - height_start) // height_interp):
    for k in range(1):
        height = height_start + height_interp * k
        img = np.zeros((frames, width_end - width_start), np.uint8)
        for i in range(frame_start, frame_start + frames):
            img_fp = img_folder + '/%d.jpg' % i
            img_temp = cv2.cvtColor(cv2.imdecode(np.fromfile(img_folder + '/%d.jpg' % i, dtype=np.uint8), -1),
                                    cv2.COLOR_BGR2GRAY)
            img[i - frame_start, :] = np.flip(img_temp[height, width_start: width_end])
        if frames == width_end - width_start:
            M = cv2.getRotationMatrix2D((frames // 2, frames // 2), rotate_angle, 1.0)
            img_rotate = cv2.warpAffine(img, M, (frames, frames))
            interp = int(np.ceil(frames * (1 - np.sqrt(2) / 2)))
            img = img_rotate[interp: frames - interp, :]
        imgs.append(img)
    return imgs


if __name__ == '__main__':
    img_folder = 'D:\实验室相关\huawei-flow-detection-code\data\完备数据集\护校河\图片集'
    img_num = len(glob.glob(os.path.join(img_folder, '*.jpg')))
    bgt_velocities = np.loadtxt('bgt_result.txt')
    sta_velocities = []
    calTextureAngle = calTextureAngleSTA
    rotate_angle = -45
    for i in range(2000, 2300):
        print('generate sti img started from %d' % i)
        sti = generate_sti(100, 400, 30, 2200, 2400, 200, i, rotate_angle=rotate_angle)
        alphas = list([calTextureAngle(img) - rotate_angle - 4 for img in sti])
        for index, alpha in enumerate(alphas):
            alphas[index] = alpha if alpha < 90 else 180 - alpha
        alpha = sum(alphas) / len(alphas)
        sta_velocities.append(np.tan(alpha / 180 * np.pi) * 0.015)

    np.savetxt('sta_result.txt', np.array(sta_velocities))

    plt.plot(sta_velocities, 'r')
    plt.plot(bgt_velocities, 'b')
    plt.xlabel('No.img')
    plt.ylabel('velocity/(mm/s)')
    plt.grid(True)
    plt.legend(['bgt', 'sta'])
    plt.show()

# img1 = generate_sti(400, 1800, 2000, 90, 2000)
# cv2.imshow('test', img1)
# cv2.waitKey(0)
