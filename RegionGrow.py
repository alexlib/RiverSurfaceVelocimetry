import numpy as np
import os
import cv2
import time


def regionGrow(img_path, borderNum=4, seedGenerator=None, criterion=None, threshold=10):
    """
    :param img_path: image filepath
    :param borderNum: 4 or 8
    :param seedGenerator: a callable function for generating seeds with input as image and output as
    a list of seeds, default None
    :param criterion: a callable function for calculating similarity with input as img, seed point, candidate points and
    output as a ndarray filled with true or false, default None
    :return: an image mask where the interested points are 1 and others are 0
    """
    assert os.path.exists(img_path) and os.path.isfile(img_path), 'illegal image filepath'
    assert borderNum == 4 or borderNum == 8, 'number of border must be 4 or 8'

    img = cv2.imread(img_path)
    img = img.astype(int)
    assert len(img.shape) == 3, 'only three channels images are allowed'
    h, w = img.shape[:2]
    if borderNum == 4:
        offsets = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    else:
        offsets = [[-1, -1], [-1, 0], [-1, 1], [0, 1], [0, -1], [1, 1], [1, 0], [1, -1]]
    if seedGenerator is not None:
        seeds = seedGenerator(img)
    else:
        seeds = [[int(h / 2), int(w / 2)], [int(h / 2), int(w * 3 / 4)],
                 [int(h * 3 / 4), int(w / 2)], [int(h * 3 / 4), int(w * 3 / 4)]]
    mask = np.zeros((h, w), dtype=np.int)
    while len(seeds) != 0:
        for seed in seeds:
            mask[tuple(seed)] = 1
            borders = list([[seed[0] + offset[0], seed[1] + offset[1]] for offset in offsets])
            borders = list([border for border in borders if 0 < border[0] < h and 0 < border[1] < w
                            and mask[tuple(border)] != 1])
            if criterion is not None:
                flag = criterion(img, seed, borders)
                indice = list([tuple(borders[index] for index in np.where(flag == True)[0])])
            else:
                diff = np.array(list([sum(np.abs(img[border[0], border[1], :] - img[seed[0], seed[1], :]))
                                      for border in borders]))
                indice = list([tuple(borders[index]) for index in np.where(diff < threshold)[0]])
            seeds.remove(seed)
            if len(indice) != 0:
                mask[tuple(np.transpose(indice))] = 1
            seeds = seeds + indice

        # mask[tuple(np.transpose(seeds))] = 1
        # borders = list(list([[seed[0] + offset[0], seed[1] + offset[1]] for offset in offsets
        #                      if 0 < seed[0] + offset[0] < h and 0 < seed[1] + offset[1] < w and
        #                      mask[seed[0] + offset[0], seed[1] + offset[1]] != 1]) for seed in seeds)
        # diff = list([list([np.sum(np.abs(img[border[0], border[1], :] - img[pair[0][0], pair[0][1], :]))
        #                    for border in pair[1]]) for pair in zip(seeds, borders)])
        # indice = list([list([pair[1][index] for index in np.where(np.array(pair[0]) < threshold)[0]])
        #                for pair in zip(diff, borders)])
        # indice = sum(indice, [])
        # if len(indice) != 0:
        #     mask[tuple(np.transpose(indice))] = 1
        # seeds = indice
        # print(len(seeds))

    mask = mask.astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    for channel in range(img.shape[2]):
        img[..., channel] = img[..., channel] * mask
    return img


if __name__ == '__main__':
    img_path = 'd:/1.png'
    borderNum, threshold = 8, 10
    seedGenerator, criterion = None, None
    start_time = time.time()
    processed_img = regionGrow(img_path, borderNum, seedGenerator, criterion, threshold)
    img = cv2.imread(img_path)
    show_img = np.concatenate([img, processed_img], axis=1)
    show_img = show_img.astype(np.uint8)
    end_time = time.time()
    print('Processing time: %.2fs' % (end_time - start_time))
    cv2.imshow('result', show_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
