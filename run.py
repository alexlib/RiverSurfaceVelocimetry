from BGTMethod import *
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error

if __name__ == '__main__':
    start = time.time()

    label = np.loadtxt('speed_label.txt')
    label_cal = np.zeros((label.shape[0], 1), dtype=np.float)
    err_cal = label_cal
    for i in range(label.shape[0]):
        img = cv2.imread('12.4 stiv images/%d_%d_599.jpg' % (i, 90 + i))
        label_cal[i, 0] = np.math.tan(calTextureAngle(img, 5, [2, 2], "WB-BGT")) * 0.124665
        err_cal[i, 0] = abs(1 - label_cal[i, 0] / label[i, 0])
        print('processing img %d' % i)

    end = time.time()

    err_cal_smooth = err_cal.copy()
    for i in range(err_cal_smooth.shape[0]):
        if err_cal_smooth[i, 0] > 1:
            err_cal_smooth[i, 0] = 1

    plt.figure()
    plt.plot(np.arange(label.shape[0]), label[:, 0], 'r')
    plt.plot(np.arange(label.shape[0]), label_cal[:, 0], 'b')
    plt.xlabel('frame')
    plt.ylabel('v(m/s)')
    plt.legend(['real', 'BGT'])
    plt.grid(True)

    plt.figure()
    plt.plot(np.arange(label.shape[0]), err_cal[:, 0], 'r')
    plt.xlabel('frame')
    plt.ylabel('err_v(m/s)')
    plt.grid(True)

    plt.figure()
    plt.plot(np.arange(label.shape[0]), err_cal_smooth[:, 0], 'r')
    plt.xlabel('frame')
    plt.ylabel('err_v_smooth(m/s)')
    plt.grid(True)

    plt.show()

    rmse = mean_squared_error(label[:, 0], label_cal)
    print('rmse of BGT in 12.4 stiv images is %.5e, cal cost time: %fs' % (rmse, end - start))