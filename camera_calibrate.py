import numpy as np
import os
import cv2


def generatePositions(rows, cols):
    """
    为 calibrateCamera 创建真实世界坐标系

    :param rows: 棋盘格行数
    :param cols: 棋盘格列数
    :return: [row * col个, (X, Y, 0)]
    """
    total = rows * cols
    object_points = np.zeros([total, 3], dtype=np.float32)
    count = 0

    for i in range(cols):
        for j in range(rows):
            object_points[count][0] = j
            object_points[count][1] = i
            count += 1

    return object_points


def getImgCorners(img, rows, cols, browser, browse_interval):
    """
    自动检测图片中的棋盘角点，并输出
    :param img: 图片矩阵(BGR 3channels)
    :param rows: 棋盘格行数
    :param cols: 棋盘格列数
    :param browser: 是否查看被检测到的棋盘点, 如果True则会弹出框显示
    :param browse_interval: 查看间隔,单位为ms, -1则会冻住直到按下一个按键
    :return: (ret, corners) 如果检测成功ret为True, 并且corners的结构为[rows * cols 个, (u, v)]
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(img, (rows, cols))
    if browser:
        cv2.namedWindow('chessboard', cv2.WINDOW_NORMAL)
        cv2.imshow('chessboard', cv2.drawChessboardCorners(img, (rows, cols), corners, None))
        cv2.waitKey(browse_interval)
    return ret, corners


def calibrateCamera(img_folder_path, rows, cols, scale=1, browser=False, browse_interval=100, result_save_path=None):
    """
    标定相机，并把相机内参和畸变系数输出为txt
    :param img_folder_path: 包含有棋盘图的文件夹路径
    :param rows: 棋盘的行数
    :param cols: 棋盘的列数
    :param scale: 图片缩放尺度, 默认为1
    :param browser: 是否查看棋盘格检测的过程
    :param browse_interval: 查看棋盘格检测的间隔
    :param result_save_path: 内参和畸变系数保存的文件夹路径
    :return: None, 内参保存为 result_save_path/intrinsic.txt 畸变参数保存为 result_save_path/distcoef.txt
    """
    WIDTH = 0
    HEIGHT = 0

    img_files = os.listdir(img_folder_path)

    all_object_points = []
    all_img_points = []

    for each_img_file in img_files:
        if not each_img_file.endswith('.jpg'):
            continue

        # 对于每张图片，获取其棋盘标定点的(u,v) 和 (X,Y,Z)

        object_positions = generatePositions(rows=rows, cols=cols)
        img_file_name = img_folder_path + '/' + each_img_file
        this_image = cv2.imread(img_file_name)

        if WIDTH == 0:
            WIDTH = int(this_image.shape[1] * scale)
            HEIGHT = int(this_image.shape[0] * scale)

        if scale != 1:
            this_image = cv2.resize(this_image, (WIDTH, HEIGHT))

        ret, corners = getImgCorners(this_image, rows, cols, browser, browse_interval)

        # 如果成功获取，则放入集合中
        if ret:
            all_object_points.append(object_positions)
            all_img_points.append(corners)

    print('已加载%d个图像' % len(all_img_points))

    # 标定
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(all_object_points, all_img_points,
                                                       (WIDTH, HEIGHT), None, None)

    if result_save_path is not None:
        # 把数据存下来
        np.savetxt(os.path.join(result_save_path, 'intrinsic.txt'), mtx)
        np.savetxt(os.path.join(result_save_path, 'distcoef.txt'), dist)

    print('内参保存在已保存在 %s 文件夹下' % result_save_path)



if __name__ == '__main__':
    videos_path = r"D:\calibrateVideos"
    assert os.path.exists(videos_path) and os.path.isdir(videos_path), 'illegal video path'
    images_path = os.path.join(videos_path, 'images')
    result_path = os.path.join(videos_path, 'result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    for video_name in os.listdir(videos_path):
        if video_name.endswith('.mp4'):
            video_path = os.path.join(videos_path, video_name)
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            if ret:
                image_path = os.path.join(images_path, video_name.replace('.mp4', '.jpg'))
                cv2.imwrite(image_path, frame)
            cap.release()

    row, col, scale = 5, 8, 1
    browser, browse_interval = False, -1
    result_save_path = result_path
    calibrateCamera(images_path, row, col, scale, browser, browse_interval, result_save_path)
