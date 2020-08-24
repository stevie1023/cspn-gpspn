import h5py
import sys
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from scipy import signal
import pandas as pd
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
        hf = h5py.File("/export/homebrick/home/mzhu/madesi/data/IceDetection_ieckai10.h5", "r")
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
        n_i = data.get('_0.0-0.0-4.0')
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



def load_data(file_name, data_flag, channel, is_plot=False):
    # channel_name = get_channel_name(channel)

    # load data
    ts_raw = data_read_from_file(file_name, data_flag)
    # # normalize data
    # ts_norm = data_normalize(ts_raw)
    features = [8,9,10,11,12,13]
    ts_raw = ts_raw[3000:8001, features]


    # down sample
    # sample_step = 25
    # l_start = 15000
    # use_mean = True


    # # sliding window to get slice of data
    # window_size = 32
    # step_size = 1
    # ts_split = data_split_sliding_window(ts_sample, window_size, step_size)
    #
    # ts_fft = data_fft(ts_split)

    # plot
    # if is_plot:
    #     data_plot(ts_raw, channel, channel_name+'_raw')
    #     data_plot(ts_norm, channel, channel_name+'_norm')
    #     data_plot(ts_sample, channel, channel_name+'_sample')
        # data_plot(ts_split[:, :, 4], channel, channel_name+'_window')

    return ts_raw
    # return ts_fft



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
    channels_ =[]
    channels_selected = [7,8,9,14,15,20,21,22,23,24,25,26]
    for i in channels_selected:
        channels_.append(channels[i])
    return channels_





if __name__ == '__main__':

    ts_raw = load_data(None, None, 8 , True)
    data1 = pd.DataFrame(ts_raw)
    data1.to_csv('data6dabnormal.csv',header = False, index = False)

    print('done')