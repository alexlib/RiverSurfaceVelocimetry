# slambook2 dense mapping in python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

# parameter
boarder = 20  # 边缘宽度
width, height = 640, 480  # 图像宽度，高度
fx, fy = 481.2, -480.0
cx, cy = 319.5, 239.5  # 相机内参
ncc_window_size = 3  # NCC窗口半宽度
ncc_area = (2 * ncc_window_size + 1) ** 2  # NCC窗口面积
min_cov, max_cov = 0.1, 10  # 收敛/发散判定方差


def main():
    fp = ''
    if not os.path.exists(fp):
        print('Reading image files failed')
    else:
        return 0


if __name__ == '__main__':
    main()
