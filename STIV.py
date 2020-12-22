import cv2 as cv
import numpy as np
import ImgUtil
import math
from scipy.integrate import simps
import threading
import cv2


def getjxx(matrix1, matrix2):
    jxxs = []
    for i in range(matrix1.shape[0]):
        jxx_temp = 0
        for j in range(1, matrix1.shape[1]):
            jxx_temp += (matrix1[i, j] * matrix2[i, j] + matrix1[i, j - 1] * matrix2[i, j - 1]) / 2
        jxxs.append(jxx_temp)
    jxx = sum(jxxs) - (jxxs[0] + jxxs[len(jxxs) - 1] / 2)
    return jxx


def getJ(mat1, mat2):
    assert mat1.shape[0] == mat2.shape[0]
    res = np.sum(simps(mat1) * simps(mat2))
    return res


def solveAngle(img):
    # round_x = np.zeros((img.shape[0] - 6, img.shape[1] - 6))
    # round_t = np.zeros((img.shape[0] - 6, img.shape[1] - 6))
    # for i in range(round_x.shape[0]):
    #     for j in range(round_x.shape[1]):
    #         round_x[i, j] = (img[i, j + 3] - 9 * img[i, j + 2] + 45 * (img[i, j + 1] - img[i, j - 1]) - 9 * img[i, j - 2] - img[i, j - 3]) / 60
    #         round_t[i, j] = (img[i + 3, j] - 9 * img[i + 2, j] + 45 * (img[i + 1, j] - img[i - 1, j]) - 9 * img[i - 2, j] - img[i - 3, j]) / 60

    round_x = np.zeros((img.shape[0], img.shape[1]))
    round_t = np.zeros((img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if 0 <= i - 2 and i + 2 < img.shape[0]:
                round_t[i, j] = (img[i + 2, j] - 8 * img[i + 1, j] + 8 * img[i - 1, j] - img[i - 2, j]) / 12
            if 0 <= j - 2 and j + 2 < img.shape[1]:
                round_x[i, j] = (img[i, j + 2] - 8 * img[i, j + 1] + 8 * img[i, j - 1] - img[i, j - 2]) / 12

    round_x = round_x[2:-2, 2:-2]
    round_t = round_t[2:-2, 2:-2]

    jxx = getjxx(round_x, round_x)
    jxt = getjxx(round_x, round_t)
    jtt = getjxx(round_t, round_t)

    clear = np.sqrt((jxx - jtt) ** 2 + 4 * jxt ** 2) / (jxx + jtt)

    print('C值为\t' + str(clear))

    tan2fai = 2 * jxt / (jtt - jxx)
    fai = math.atan(tan2fai) / math.pi * 90
    return fai


def multipleSTIV(img_folder, height_start, height_end, width_start, width_end, frame_start, frame_end, line_width=1,
                 img_output_folder=None, drawline_only=False):
    threads = []
    imgs = ImgUtil.getImgListArray(img_folder, frame_start, frame_end)

    if drawline_only:
        hud = imgs[0]
        for each_height in range(height_start, height_end, 2):
            hud = cv.line(hud, (width_start, each_height), (width_end, each_height), (0, 0, 255), 1)
        cv.imshow('img', hud)
        cv.waitKey(0)
        return

    for each_height in range(height_start, height_end):
        img_output_path = img_output_folder + '/' + str(frame_start) + '_' + str(frame_end) + '_' + str(
            each_height) + '.jpg' if img_output_folder is not None else None
        this_thread = threading.Thread(target=singleSTIV, args=(imgs, each_height, width_start,
                                                                width_end, line_width, img_output_path))
        threads.append(this_thread)

    for each_thread in threads:
        each_thread.start()

    for each_thread in threads:
        each_thread.join()

    print('finished')


def singleSTIV(imgs, height, width_start, width_end, line_width=1, img_output_path=None):
    rows = int(len(imgs) * line_width)
    cols = int(width_end - width_start)
    this_img = np.zeros([rows, cols, 3], dtype=np.uint8)

    for i in range(int(rows / line_width)):
        this_img[i * line_width:(i + 1) * line_width] = imgs[i, height:height + line_width, width_start:width_end, :]

    # this_line = imgs[:, height:height+line_width, width_start:width_end, :]
    this_line = this_img

    grey = cv.cvtColor(this_line, cv.COLOR_BGR2GRAY)

    if img_output_path is not None:
        cv.imwrite(img_output_path, grey)
    # cv.imshow('img', grey)
    # cv.waitKey(0)


def main():
    img_folder = 'images'
    img_output_folder = 'sti'

    for each_frame_start in range(0, 1800):
        multipleSTIV(img_folder=img_folder, height_start=680, height_end=700, width_start=500, width_end=1000,
                     frame_start=each_frame_start, frame_end=each_frame_start + 90, line_width=1,
                     img_output_folder=img_output_folder, drawline_only=True)


if __name__ == '__main__':
    main()
