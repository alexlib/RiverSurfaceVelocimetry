import numpy as np
import cv2
import scipy.io as scio
from math import cos, atan, radians, degrees, tan, sin

fp_img = r"D:\QQFiles\2485246728\FileRecv\MobileFile\1615351867304.jpg"

# 依次是Matlab标定得到的内参、径向畸变参数和切向畸变参数
intrinsicMatrix = scio.loadmat('intrinsicMatrix.mat')['intrinsicMatrix']
RadialDistortion = scio.loadmat('radialDistortion.mat')['radialDistortion']
TangentialDistortion = scio.loadmat('tangentialDistortion.mat')['tangentialDistortion']

# 将其转变为OpenCV能接受的形式
cameraMatrix = intrinsicMatrix.T
distcoeffs = np.array([RadialDistortion[0, 0], RadialDistortion[0, 1],
                       TangentialDistortion[0, 0], TangentialDistortion[0, 1],
                       RadialDistortion[0, 2]], dtype=np.float64)

# objectPoints、imagePoints、rvec、tvec依次为世界坐标、像素坐标、旋转矩阵、平移向量
# 上下两簇分别代表元水位和升高后的场景
objectPoints = np.array([[-100, 100, 0], [100, 100, 0], [100, -100, 0], [-100, -100, 0]], dtype=np.float64)
imagePoints = np.array([[2656, 787], [3161, 765], [3204, 1100], [2663, 1121]], dtype=np.float64)
retval, rvec, tvec = cv2.solvePnP(objectPoints, imagePoints, cameraMatrix, distcoeffs)
# solvePnP出来的rvec是向量，需要转换下，下同
rvec = cv2.Rodrigues(rvec)[0]

imagePoints1 = np.array([[2589, 225], [3165, 197], [3217, 525], [2590, 545]], dtype=np.float64)
retval, rvec1, tvec1 = cv2.solvePnP(objectPoints, imagePoints1, cameraMatrix, distcoeffs)
rvec1 = cv2.Rodrigues(rvec1)[0]

# Zc1、Zc2代表原图的四个点的计算值Zc，Zc3、Zc4代表升高后的四个点的计算值Zc
# 其中Zc1中记录世界坐标直接转变成相机坐标的Zc，Zc2中采取与像素坐标转换后相除的Zc，并取Xc和Yc处的均值
# Zc3 和 Zc4 同 Zc1 和 Zc2
testObjPoints = np.zeros((16, 3), dtype=np.float64)
testObjPoints[0: 4, :] = objectPoints
testObjPoints[4: 16, :] = np.array([[0, 100, 0], [200, 100, 0], [-100, 0, 0], [0, 0, 0],
                                 [100, 0, 0], [200, 0, 0], [0, -100, 0], [200, -100, 0],
                                 [-100, -200, 0], [0, -200, 0], [100, -200, 0], [200, -200, 0]], dtype=np.float64)
Zc1, Zc2 = [], []
for i in range(testObjPoints.shape[0]):
    Zc1.append((rvec @ np.reshape(testObjPoints[i, :], tvec.shape) + tvec)[2, 0])
    # imgPointCal = cameraMatrix @ (rvec @ np.reshape(testObjPoints[i, :], tvec.shape) + tvec)
    # Zc2.append(sum(imgPointCal[0:2, 0] / imagePoints[i]) / 2)

testImagePoints = np.zeros((16, 2), dtype=np.float64)
testImagePoints[0: 4, :] = imagePoints
testImagePoints[4: 16, :] = np.array([[2908, 775], [3414, 753], [2660, 948], [2920, 937], [3182, 936], [3441, 914],
                                      [2934, 1111], [3474, 1087], [2668, 1306], [2947, 1297], [3226, 1283], [3507, 1268]])

# Zc3, Zc4 = [], []
# for i in range(objectPoints.shape[0]):
#     Zc3.append((rvec1 @ np.reshape(objectPoints[i, :], tvec1.shape) + tvec1)[2, 0])
#     imgPointCal = cameraMatrix @ (rvec1 @ np.reshape(objectPoints[i, :], tvec1.shape) + tvec1)
#     Zc4.append(sum(imgPointCal[0:2, 0] / imagePoints1[i]) / 2)


# 测试下旋转矩阵和平移向量是否求解正确，index是选哪个点进行计算
# 上下两簇分别测试原图和升高后的图
# index = 1
# objPoint = np.linalg.inv(rvec) @ (Zc1[index] * np.linalg.inv(cameraMatrix) @
#                     np.reshape(np.array(list(imagePoints[index]) + [1]), tvec.shape) - tvec)
# print(objPoint)
#
# index = 1
# objPoint = np.linalg.inv(rvec1) @ (Zc3[index] * np.linalg.inv(cameraMatrix) @
#                     np.reshape(np.array(list(imagePoints1[index]) + [1]), tvec1.shape) - tvec1)
# print(objPoint)



res = []
# h、dh、alpha、index分别是相机距地面高度、水面升高高度、相机拍摄俯仰角和测试点
h, dh, alpha = 1024, 246, 35.4
for index in range(len(Zc1)):
    # 两种情况下进行水面升高后的zc计算，计算公式参考PDF
    if Zc1[index] > h / cos(radians(alpha)):
        beta = degrees(atan(((Zc1[index] - h / cos(radians(alpha)))
                             / sin(radians(alpha)) + h * tan(radians(alpha))) / (h - dh))) - alpha
        zc1 = ((Zc1[index] - h / cos(radians(alpha))) / sin(radians(alpha)) + h * tan(radians(alpha))) \
              / sin(radians(alpha + beta)) * cos(radians(beta))
    else:
        beta = degrees(atan((h * tan(radians(alpha)) - (h / cos(radians(alpha)) - Zc1[index])
                             / sin(radians(alpha))) / (h - dh)))
        zc1 = (h - dh) / cos(radians(beta)) * cos(alpha - beta)
    objPoint = np.linalg.inv(rvec) @ (zc1 * np.linalg.inv(cameraMatrix) @
                                      np.reshape(np.array(list(testImagePoints[index]) + [1]), tvec.shape) - tvec)
    # 因为世界坐标选在原世界坐标系下，所以要减掉高度差
    objPoint[2, 0] = objPoint[2, 0] - dh
    # 输出结果，查看是否与对应ObjectPoints[index]相同
    res.append((testObjPoints[index] - objPoint)[0:2, 0])
with open('d:/res.txt', 'w', encoding='utf8') as f:
    for i in range(len(res)):
        f.write(str(res[i]) + '\n')
