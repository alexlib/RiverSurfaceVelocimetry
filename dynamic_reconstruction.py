import cv2
import numpy as np
import os
from math import fabs
import matplotlib.pyplot as plt


def getImgPoints(img_point_file_path, img=None, show_img=False):
    """
    根据img_points.txt，把标记点加载进来，并返回
    :param img_point_file_path: 存放img_points的txt文件, 格式为 'u0 v0\nu1 v1...'
    :param img: 图片矩阵, 为了做展示用, 默认为空
    :param show_img: 是否把读取到的标记点展示到图片上, 默认否
    :return: 像素点位置的集合 [个数, (u, v)]
    """
    img_points = np.loadtxt(img_point_file_path)
    img_points = img_points.astype(np.int)
    if show_img and img is not None:
        new_img = img.copy()
        for i in range(len(img_points)):
            new_img = cv2.putText(new_img, str(i), (img_points[i][0], img_points[i][1]), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (0, 0, 0), 3)
            cv2.drawMarker(new_img, (img_points[i][0], img_points[i][1]), (0, 0, 255), 2, 20)
        cv2.namedWindow('getImgPoints', cv2.WINDOW_NORMAL)
        cv2.imshow('getImgPoints', new_img)
        cv2.waitKey(0)
    return img_points.astype(np.float64)


def getObjectPoints(object_point_file_path, z=0):
    """
    根据object_points.txt，把标记点加载进来，并返回
    :param object_point_file_path: 存放object_points的txt文件, 格式为 'X0 Y0\nX1 Y1...'
    :param z: 平面的z坐标
    :return: output [个数, (X, Y, Z)]
    """

    object_points = np.loadtxt(object_point_file_path)
    output = np.zeros([object_points.shape[0], 3])
    output[:, :2] = object_points
    output[:, 2] = z
    return output


def getImagePosByObjectPos(X, Y, Z, intrinsic_mat, rotation_mat, translation_vec):
    """
    根据世界坐标系, 获得对应图片的u,v坐标
    :param X:
    :param Y:
    :param Z:
    :param intrinsic_mat: 内参
    :param rotation_mat: 旋转矩阵
    :param translation_vec: 平移矩阵
    :return: u,v (int)
    """
    XYZ = np.array([X, Y, Z]).reshape(3, 1)
    translation_vec = translation_vec.reshape(3, 1)
    xy1 = intrinsic_mat @ (rotation_mat @ XYZ + translation_vec)
    xy1 = xy1 / xy1[2, 0]
    return int(xy1[0]), int(xy1[1])


def getObjectPosByImagePos(u, v, Z, intrinsic_mat, rotation_mat, translation_vec):
    '''
    根据u,v和当前高度，得到真实坐标
    :param u:
    :param v:
    :param Z:
    :param intrinsic_mat: 内参
    :param rotation_mat: 旋转矩阵
    :param translation_vec: 平移矩阵
    :return: [X,Y]
    '''
    translation_vec = translation_vec.reshape(3, 1)
    A = intrinsic_mat @ rotation_mat
    B = intrinsic_mat @ translation_vec

    C = np.zeros([2, 2])
    D = np.zeros([2, 1])

    C[0, 0] = A[2, 0] * u - A[0, 0]
    C[0, 1] = A[2, 1] * u - A[0, 1]
    C[1, 0] = A[2, 0] * v - A[1, 0]
    C[1, 1] = A[2, 1] * v - A[1, 1]

    D[0] = (A[0, 2] - A[2, 2] * u) * Z + B[0] - B[2] * u
    D[1] = (A[1, 2] - A[2, 2] * v) * Z + B[1] - B[2] * v

    XY = np.linalg.solve(C, D)
    return np.squeeze(XY)


if __name__ == '__main__':
    img_point_file_path = r"D:\calibrateVideos\img_points.txt"
    object_point_file_path = r"D:\calibrateVideos\object_points.txt"
    img_points = getImgPoints(img_point_file_path)
    object_points = getObjectPoints(object_point_file_path)
    intrinsic_fp = r"D:\calibrateVideos\result\intrinsic.txt"
    distcoef_fp = r"D:\calibrateVideos\result\distcoef.txt"
    intrinsic_mat = np.loadtxt(intrinsic_fp)
    distcoef_mat = np.loadtxt(distcoef_fp)
    retval, rvec, tvec = cv2.solvePnP(object_points, img_points, intrinsic_mat, distcoef_mat)
    rmat = cv2.Rodrigues(rvec)[0]
    loss = []
    for i in range(object_points.shape[0]):
        x, y = getImagePosByObjectPos(object_points[i][0], object_points[i][1], object_points[i][2],
                                      intrinsic_mat, rmat, tvec)
        loss.append([fabs(x - img_points[i][0]), fabs(y - img_points[i][1])])
    loss = np.array(loss)
    plt.figure()
    plt.plot(loss[:, 0])
    plt.plot(loss[:, 1])
    plt.xlabel('No.number')
    plt.ylabel('Reconstruction Error(pixel)')
    plt.legend(['x', 'y'])
    plt.grid(True)
    plt.show()
