import cv2 as cv
import os
import numpy as np


def get_img(image_id):
    dest_path = './data/' + str(image_id) + '.jpg'
    return cv.imread(dest_path)


def show_img(image, title='image', wait_time = 0):
    cv.imshow(title, image)
    cv.waitKey(wait_time)


def video2img(src_path, dest_path, just_one=False, rotation=None, skip_count=0):
    # 捕获视频
    cap = cv.VideoCapture(src_path)

    while skip_count > 0:
        skip_count -= 1
        cap.read()

    # 设置图片编号，从0开始
    img_id = 0

    # 循环保存图片
    while cap.isOpened():

        # 读取一帧图片
        ret, frame = cap.read()

        if (rotation is not None):
            frame = cv.rotate(frame, rotation)

        # 如果图片是空跳出
        if not ret:
            break

        # cv.imshow('frame', frame)

        # 给图片命名
        img_name = str(img_id) + '.jpg'

        # 设置好输出路径
        output_path = os.path.join(dest_path, img_name)

        # 输出
        cv.imwrite(output_path, frame)

        print(output_path)

        if just_one:
            break

        # 编号递增
        img_id += 1

    # 释放捕获器
    cap.release()
    cv.destroyAllWindows()


def bgr2gray(image):
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def getImgList(img_folder):
    img_files = os.listdir(img_folder)
    imgs = []
    for each_img_file in img_files:
        this_img = cv.imread(img_folder + '/' + each_img_file)
        imgs.append(this_img)
    return imgs


def getImgListArray(img_folder, frame_start=None, frame_end=None):
    img_files = os.listdir(img_folder)

    img_shape = cv.imread(img_folder + '/' + img_files[0]).shape

    width = img_shape[1]
    height = img_shape[0]
    channel = img_shape[2]

    if frame_start is None:
        imgs = np.zeros([len(img_files), height, width, channel], dtype=np.uint8)
        for index in range(len(img_files)):
            this_img_path = img_folder + '/' + str(index + 1) + '.jpg'
            this_img = cv.imread(this_img_path)
            imgs[index] = this_img
            cv.waitKey(0)

    else:
        imgs = np.zeros([frame_end - frame_start, height, width, channel], dtype=np.uint8)
        img_index = 0
        for index in range(frame_start, frame_end):
            this_img_path = img_folder + '/' + str(index) + '.jpg'
            this_img = cv.imread(this_img_path)
            # print('加载了%s为imgs[%d]' % (this_img_path, img_index))
            imgs[img_index] = this_img
            img_index += 1

    return imgs


def imgs2video(img_path, output_path, fps, pic_num, img_size, alter_size=False):
    # 图片路径
    im_dir = img_path
    # 输出视频路径
    video_dir = output_path
    # 帧率
    fps = fps
    # 图片数
    num = pic_num

    fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv.VideoWriter(video_dir, fourcc, fps, img_size)

    for i in range(1, num):
        im_name = os.path.join(im_dir, str(i) + '.jpg')
        frame = cv.imread(im_name)

        assert frame.shape[0] > 0 and frame.shape[1] > 0

        if alter_size:
            frame = cv.resize(frame, img_size)
        video_writer.write(frame)
        print('process\t' + str(i))
        # cv.imshow('rr', frame)
        # cv.waitKey(1)

    video_writer.release()
    print('finish')


if __name__ == '__main__':
    # src_path = '../data/laser move 0_1.avi'
    # dest_path = '../data/fuck/'
    # video2img(src_path=src_path, dest_path=dest_path, skip_count=0)



    img_path = '../data/transform 0.1'
    imgs2video(img_path=img_path, output_path='../data/laser move 0.1.avi', fps=15, pic_num=49, img_size=(3200, 1800),
               alter_size=True)

    print('finished')