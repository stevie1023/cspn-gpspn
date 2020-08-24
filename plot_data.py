import h5py
import numpy as np
import matplotlib.pyplot as plt


def plot_channels(ts, t_start, t_end):
    k1=t_start
    k2=t_end
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        col = i
        plt.plot(np.arange(k1, k2), ts[k1:k2, col])
    # plt.savefig('./data_plot1.pdf')
    plt.show()

    plt.figure(figsize=(10,10))
    for i in range(9, 18):
        plt.subplot(3, 3, i - 8)
        col = i
        plt.plot(np.arange(k1, k2), ts[k1:k2, col])
    # plt.savefig('./data_plot2.pdf')
    plt.show()

    plt.figure(figsize=(10,10))
    for i in range(18, 27):
        plt.subplot(3, 3, i - 17)
        col = i
        plt.plot(np.arange(k1, k2), ts[k1:k2, col])
    # plt.savefig('./data_plot3.pdf')
    plt.show()


if __name__ == '__main__':

    # load file
    hf_old1 = h5py.File("/export/homebrick/home/mzhu/madesi/data/IceDetection.h5", "r")
    # hf = h5py.File("/media/yu/data/yu/dataset/madesi/Simulations/AnomalyDetection_Masschange.h5", "r")
    hf = h5py.File("/export/homebrick/home/mzhu/madesi/data/IceDetection.h5", "r")
    # show attributes
    print(hf.keys())

    # # print data attributes for all keys
    # data_temp = hf.get('base_model')
    # print(data_temp.keys())
    # data_temp = hf.get('dim0')
    # print(data_temp.keys())
    # data_temp = hf.get('dim1')
    # print(data_temp.keys())


    # get data from attribute 'data'
    data = hf.get('data')

    # show data attributes
    print(data.keys())


    k1 = 3000
    k2 = 6400
    # get series from data
    for data_label in data.keys():
        n_i = data.get(data_label)
        ts = np.array(n_i)
        print(ts.shape)

        plt.figure(figsize=(10,10))
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            col = i
            plt.plot(np.arange(k1, k2), ts[k1:k2, col])
        plt.savefig('./data_plot1.pdf')
        # plt.show()

        plt.figure(figsize=(10,10))
        for i in range(9, 18):
            plt.subplot(3, 3, i - 8)
            col = i
            plt.plot(np.arange(k1, k2), ts[k1:k2, col])
        plt.savefig('./data_plot2.pdf')
        # plt.show()

        plt.figure(figsize=(10,10))
        for i in range(18, 27):
            plt.subplot(3, 3, i - 17)
            col = i
            plt.plot(np.arange(k1, k2), ts[k1:k2, col])
            break
        plt.savefig('./data_plot3.pdf')
        # plt.show()

        # split and save data.


    print('done')
