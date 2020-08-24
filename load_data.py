import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import signal

from plot_data import plot_channels


def data_read_from_file(file_name=None, data_flag=None):
    """
    load h5 file
    :param: file_name: file to load data
            data_flag: 0 - normal, no weight; 1 - weight added
    :return: a matrix with each column a TS
    """
    if file_name is None:
        # load file
        # hf = h5py.File("/media/yu/data/yu/dataset/madesi/Simulations/Masschange_UniformWind.h5", "r")
        # hf = h5py.File("/media/yu/data/yu/dataset/madesi/Simulations/AnomalyDetection_Masschange.h5", "r")
        hf = h5py.File("/export/homebrick/home/mzhu/madesi/data/IceDetection.h5", "r")
        # hf = h5py.File("/media/yu/data/yu/dataset/madesi/Imbalance.h5", "r")
    else:
        hf = h5py.File("/export/homebrick/home/mzhu/madesi/data/" + file_name, "r")
    # show attributes
    print(hf.keys())
    """
    get normal data from IceDetection.h5
    """
    # get data from attribute 'data'
    data = hf.get('data')
    print(data.keys())
    # get series from data
    if data_flag is None:
        n_i = data.get('_0.0-0.0-0.0')
    else:
        n_i = data.get(data_flag)
    ts = np.array(n_i)
    print(ts.shape)

    return ts


def data_normalize(ts):
    """
    Normalize data
    """
    miu = np.mean(ts, axis=0)
    std = np.std(ts, axis=0)
    ts_n = (ts - miu) / std
    # "time" do not need to be normalized
    t = ts[:, 0]
    ts_n[:, 0] = t

    return ts_n


def data_downsample(data_slice, sample_step, l_start=2400, use_mean=True):
    """
    cut the initial part and down sample the TS
    we take the mean of every #s steps
    :return: down sampled TS
    """
    l, p = data_slice.shape  # length & dimension of raw data
    assert l > l_start, "l_cut too large!"
    data_cut = data_slice[l_start:, :]
    l, p = data_cut.shape  # length & dimension of crop data

    if use_mean:
        # take the mean of every #sample_step sample
        l_end = sample_step * (l // sample_step)
        data_sample = np.mean(data_cut[0:l_end, :].reshape(-1, sample_step, p), axis=1)
    else:
        # take samples of every #sample_step sample
        data_sample = data_cut[0:l:sample_step, :]

    return data_sample


def data_split_sliding_window(ts, window_size=32, step_size=8):
    """
    split TS based on sliding window
    :param ts:
    :return:
    """
    # split data based on sliding window
    # length of data
    l, w = ts.shape
    # crop from data, N = (l-window_size)//step_size
    n = (l - window_size) // step_size
    data_split = np.empty((window_size, w, n + 1))
    for i in range(n + 1):
        # window data
        data_slice = ts[i * step_size:i * step_size + window_size, :]
        print('Data slice - start:', i * step_size, ', end:', i * step_size + window_size)
        data_split[:, :, i] = data_slice
        # bdata_rfft = np.fft.rfft(data_sampled, axis=0)

    return data_split
    print(data_split)

def data_fft(ts_split):
    """
    do rfft after crop with sliding window
    :param ts_split:
    :return: real-valued fft
    """
    l, w, n = ts_split.shape
    ts_rfft = np.fft.rfft(ts_split, axis=0)

    return ts_rfft


def load_data(file_name, data_flag, channel, is_plot=False):
    channel_name = get_channel_name(channel)

    # load data
    ts_raw = data_read_from_file(file_name, data_flag)

    # normalize data
    ts_norm = data_normalize(ts_raw)

    # down sample
    sample_step = 25
    l_start = 15000
    use_mean = True
    ts_sample = data_downsample(ts_norm, sample_step, l_start, use_mean)

    # sliding window to get slice of data
    window_size = 32
    step_size = 1
    ts_split = data_split_sliding_window(ts_sample, window_size, step_size)

    ts_fft = data_fft(ts_split)
    # plot
    if is_plot:
        data_plot(ts_raw, channel, channel_name+'_raw')
        data_plot(ts_norm, channel, channel_name+'_norm')
        data_plot(ts_sample, channel, channel_name+'_sample')
        data_plot(ts_split[:, :, 4], channel, channel_name+'_window')

    return ts_fft


def data_channels(data, c_list):
    data_out = data[:, c_list, :]

    return data_out


def data_reshape(data):
    """
    reshape data from (l, w, d) to (d, T_w * w)
    :param data:
    :return:
    """
    l, w, d = data.shape
    # some dirty tricks to reshape
    data_out1 = np.concatenate([data.real, data.imag], axis=0)
    data_out2 = np.transpose(data_out1, (2, 1, 0))
    data_out = data_out2.reshape([d, -1], order='C')

    return data_out


def get_channel_name(channel):
    channels = ['time', 'Wind1VelX', 'Wind1VelY', 'BldPitch1', 'Azimuth', 'RotSpeed', 'Spn3MLxb1', 'Spn3MLyb1',
                'Spn1ALxb1', 'Spn1ALyb1', 'Spn1ALxb2', 'Spn1ALyb2', 'Spn1ALxb3', 'Spn1ALyb3',
                'Spn2ALxb1', 'Spn2ALyb1', 'Spn2ALxb2', 'Spn2ALyb2', 'Spn2ALxb3', 'Spn2ALyb3',
                'Spn3ALxb1', 'Spn3ALyb1', 'Spn3ALxb2', 'Spn3ALyb2', 'Spn3ALxb3', 'Spn3ALyb3']

    return channels[channel]


def data_plot(data, channel, channel_name=None):
    x = data[:, 0]
    y = data[:, channel]
    y2 = data[:, 4]

    plt.figure()
    plt.plot(x, y)
    # plt.plot(x, y2)
    if channel_name is not None:
        plt.title(channel_name)
        plt.savefig('./plots/' + channel_name + '.pdf')
    plt.show()


if __name__ == '__main__':

    load_data(None, 9, True)

    print('done')
