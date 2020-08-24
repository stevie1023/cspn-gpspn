"""
script developing 2d complex SPN
for synthetic sine data
"""
import numpy as np
import logging
import time
import argparse
import pickle
import scipy
import matplotlib.pyplot as plt
import sys

from scipy import stats
from load_data import load_data, data_channels, data_reshape

sys.path.append('../2dgaussian/SPFlow/src/')
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats
from spn.structure.Base import Context
from spn.algorithms.LearningWrappers import learn_parametric

# current_time=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
# logging.basicConfig(filename='/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev/whittle_spn_'+current_time+'.log', filemode='w', level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

path_base = '/export/homebrick/home/mzhu/madesi/dev_version1/'


# path_base = '/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev_aies/'


def get_save_path(args):
    if args.wspn_type == 1:
        key = 'wspn1d'
    elif args.wspn_type == 2:
        key = 'wspn_pair'
    elif args.wspn_type == 3:
        key = 'wspn2d'
    else:
        print('input spn type error')
        sys.exit()

    if args.data_type == 1:
        data = 'wind/'
    else:
        print('input data type error')
        sys.exit()

    save_path = path_base + data + key + '_' + str(
        args.n_min_slice) + '_' + str(args.threshold) + '/'

    return save_path


def get_l_rfft(ARGS):
    if ARGS.data_type == 1:
        l_rfft = 17
    elif ARGS.data_type == 2:
        l_rfft = 8
    elif ARGS.data_type == 3:
        l_rfft = 17
    elif ARGS.data_type == 4:
        l_rfft = 17
    elif ARGS.data_type == 5:
        l_rfft = 51
    else:
        print('input l_rfft error')
        sys.exit()

    return l_rfft


def learn_whittle_spn_1d(train_data, n_RV, n_min_slice=2000, init_scope=None):
    from spn.structure.leaves.parametric.Parametric import Gaussian

    # learn spn
    ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # l_rfft=None --> 1d gaussian node, is_2d does not work
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=args.threshold,
                            initial_scope=init_scope, cpus=1, l_rfft=None, is_2d=False)
    save_path = get_save_path(args)
    check_path(save_path)
    f = open(save_path + 'wspn_1d.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_1d(ARGS):
    save_path = get_save_path(ARGS)
    f = open(save_path + 'wspn_1d.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    logger.info(log_msg)
    return spn


def learn_whittle_spn_pair(train_data, n_RV, n_min_slice, init_scope=None):
    from spn.structure.leaves.parametric.Parametric import Gaussian

    # learn spn
    ds_context = Context(parametric_types=[Gaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # need to pair RVs
    # need flag for 2d?
    l_rfft = get_l_rfft(args)
    # l_rfft!=None --> 2d/pair gaussian node, is_2d=False --> pair gaussian, diagonal covariance matrix
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=ARGS.threshold,
                            initial_scope=init_scope, cpus=1, l_rfft=l_rfft, is_2d=False)
    save_path = get_save_path(args)
    check_path(save_path)
    f = open(save_path + 'wspn_pair.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_pair(ARGS, log=False):
    save_path = get_save_path(ARGS)
    f = open(save_path + 'wspn_pair.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    if log:
        logger.info(log_msg)

    return spn


def learn_whittle_spn_2d(train_data, n_RV, n_min_slice, init_scope=None):
    from spn.structure.leaves.parametric.Parametric import MultivariateGaussian

    # learn spn
    ds_context = Context(parametric_types=[MultivariateGaussian] * n_RV).add_domains(train_data)

    print('learning WSPN')
    # need to pair RVs
    # need flag for 2d?
    l_rfft = get_l_rfft(args)
    # l_rfft!=None --> 2d/pair gaussian node, is_2d=True --> pairwise gaussian, full covariance matrix
    wspn = learn_parametric(train_data, ds_context, min_instances_slice=n_min_slice, threshold=args.threshold,
                            initial_scope=init_scope, cpus=1, l_rfft=l_rfft, is_2d=True)
    save_path = get_save_path(args)
    check_path(save_path)
    f = open(save_path + 'wspn_2d.pkl', 'wb')
    pickle.dump(wspn, f)
    f.close()

    return wspn


def load_whittle_spn_2d(ARGS, log=True):
    save_path = get_save_path(ARGS)
    f = open(save_path + 'wspn_2d.pkl', 'rb')
    spn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(spn)
    print(log_msg)
    if log:
        logger.info(log_msg)

    return spn


def load_whittle_spn_res(args):
    # load res-spn, need to be modified when model changed
    log_msg = 'Have you set the latest model path?'
    print(log_msg)
    logger.info(log_msg)

    rspn_path = 'ventola/em_optimized_fuse_spn_yu_sine'
    f = open(rspn_path, 'rb')
    rspn = pickle.load(f)
    f.close()

    log_msg = get_structure_stats(rspn)
    print(log_msg)
    logger.info(log_msg)

    return rspn


def data_to_2d(data, p, L):
    # transfer data from 1d to 2d
    h, w = data.shape
    l = L // 2 + 1
    data1 = data.reshape(h * p, -1)
    data1_r = data1[:, 0:l].reshape(h * p, l, 1)
    data1_i = data1[:, l:].reshape(h * p, l, 1)
    data2 = np.concatenate([data1_r, data1_i], 2)
    data2 = data2.reshape(h, -1, 2)

    return data2


def load_data_for_wspn(data_file, data_key, args):
    if args.data_type == 1:
        log_msg = 'loading wind data'
        print(log_msg)
        # train
        data_t = load_data(file_name=data_file, data_flag='_0.0-0.0-1.2', channel=0, is_plot=False)
        # select channels to train
        c_list = [8, 9, 10, 11, 12, 13]
        data_train0 = data_channels(data_t, c_list)
        data_train0 = data_reshape(data_train0)
        data_train = data_train0[0:1000, :]
        # pos
        data_pos = data_train0[1000:, :]
        # neg
        data_n = load_data(file_name=data_file, data_flag=data_key, channel=0, is_plot=False)
        data_neg = data_channels(data_n, c_list)
        data_neg = data_reshape(data_neg)
        p = 6  # dim
        L = 32  # TS length
        n_RV = len(c_list) * 2 * ((L // 2) + 1)  # number of RVs
        scope_list = np.arange(n_RV)
        scope_temp = np.delete(scope_list, np.where(scope_list % 34 == 17))
        init_scope = list(np.delete(scope_temp, np.where(scope_temp % 34 == 33)))

    else:
        print('input data error')
        sys.exit()
    print('data done')
    return data_train, data_pos, data_neg, n_RV, p, L, init_scope


def check_path(path):
    import os
    if not os.path.exists(path):
        os.makedirs(path)


def calc_ll(wspn, data_train, data_pos, data_neg):
    # calculate LL
    log_msg = 'Log-likelihood calculating...'
    print(log_msg)
    logger.info(log_msg)

    ll_train = log_likelihood(wspn, data_train)
    ll_pos = log_likelihood(wspn, data_pos)
    ll_neg = log_likelihood(wspn, data_neg)
    log_msg = '---------median-----------'
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_train=' + str(np.median(ll_train))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_test=' + str(np.median(ll_pos))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_ood=' + str(np.median(ll_neg))
    print(log_msg)
    logger.info(log_msg)
    log_msg = '--------- mean -----------'
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_train=' + str(np.mean(ll_train))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_test=' + str(np.mean(ll_pos))
    print(log_msg)
    logger.info(log_msg)
    log_msg = 'LL_ood=' + str(np.mean(ll_neg))
    print(log_msg)
    logger.info(log_msg)

    return ll_train, ll_pos, ll_neg


def save_ll(ll1, ll2, ll3):
    save_path = get_save_path(args)
    check_path(save_path)

    np.savetxt(save_path + 'll_train.csv', ll1, delimiter=',')
    np.savetxt(save_path + 'll_pos.csv', ll2, delimiter=',')
    np.savetxt(save_path + 'll_neg.csv', ll3, delimiter=',')


def init_log(ARGS):
    current_time = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    # Creating log file
    save_path = get_save_path(ARGS)
    check_path(save_path)
    # path_base = '/media/yu/data/yu/code/gp_whittle/WhittleNetwork/dev/'
    if ARGS.train_type == 1:
        file_base = 'train_wspn_' + str(ARGS.wspn_type) + '_on_data' + str(ARGS.data_type) + '_'
    elif ARGS.train_type == 2:
        file_base = 'test_wspn' + str(ARGS.wspn_type) + '_on_data' + str(ARGS.data_type) + '_'
    else:
        file_base = 'error'
    logging.basicConfig(
        filename=save_path + file_base + current_time + '.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    return logger


def run(data_file_list, data_key_list, args):
    if args.wspn_type == 1:
        # train/load wspn 1d
        n_min_slice = args.n_min_slice
        if args.train_type == 1:
            # load data for training
            data_file = data_file_list[0]
            data_key = data_key_list[1]
            data_train, data_pos, data_neg, n_RV, p, L, init_scope = load_data_for_wspn(data_file, data_key, args)
            log_msg = 'Train WSPN 1d'
            logger.info(log_msg)
            wspn = learn_whittle_spn_1d(data_train, n_RV, n_min_slice, init_scope)
        elif args.train_type == 2:
            # load SPN
            log_msg = 'Test WSPN 1d'
            logger.info(log_msg)
            wspn = load_whittle_spn_1d(args)
            # load data for test
            for i in range(1, 10):
                data_file = data_file_list[0]
                data_key = data_key_list[i]
                data_train, data_pos, data_neg, n_RV, p, L, init_scope = load_data_for_wspn(data_file, data_key, args)
                # calculate LL and save for significance test
                [ll_train, ll_pos, ll_neg] = calc_ll(wspn, data_train, data_pos, data_neg)
                save_ll(ll_train, ll_pos, ll_neg)

    if args.wspn_type == 3:
        # train/load wspn 2d
        n_min_slice = args.n_min_slice
        if args.train_type == 1:
            # load data for training
            data_file = data_file_list[0]
            data_key = data_key_list[1]
            data_train, data_pos, data_neg, n_RV, p, L, init_scope = load_data_for_wspn(data_file, data_key, args)
            log_msg = 'Train WSPN 2d'
            logger.info(log_msg)
            wspn = learn_whittle_spn_2d(data_train, n_RV, n_min_slice, init_scope)
        elif args.train_type == 2:
            # load SPN
            log_msg = 'Test WSPN 2d'
            logger.info(log_msg)
            wspn = load_whittle_spn_2d(args)
            # load data for test
            for i in range(1, 10):
                data_file = data_file_list[0]
                data_key = data_key_list[i]
                data_train, data_pos, data_neg, n_RV, p, L, init_scope = load_data_for_wspn(data_file, data_key, args)
                # calculate LL and save for significance test
                [ll_train, ll_pos, ll_neg] = calc_ll(wspn, data_train, data_pos, data_neg)
                save_ll(ll_train, ll_pos, ll_neg)


if __name__ == '__main__':
    # set parameters
    parser = argparse.ArgumentParser()
    # Args go here
    parser.add_argument('--wspn_type', type=int, default=1,
                        help='Type of wspn, 1-1d, 2-pair, 3-2d, 4-res-spn')
    parser.add_argument('--train_type', type=int, default=1,
                        help='Type of train, 1-train, 2-test')
    parser.add_argument('--n_min_slice', type=int, default=100,
                        help='minimum size of slice.')
    parser.add_argument('--data_type', type=int, default=1,
                        help='Type of data, 1-wind')
    parser.add_argument('--threshold', type=float, default=0.7,
                        help='Threshold of splitting features')

    args, unparsed = parser.parse_known_args()

    # init logger
    logger = init_log(args)
    log_msg = '\n--wspn_type=' + str(args.wspn_type) + \
              '\n--train_type=' + str(args.train_type) + \
              '\n--n_min_slice=' + str(args.n_min_slice) + \
              '\n--data_type=' + str(args.data_type) + \
              '\n--threshold=' + str(args.threshold)
    print(log_msg)
    logger.info(log_msg)
    start_time = time.time()
    np.random.seed(20200305)

    data_file_list = ["IceDetection_ieckai10.h5", "IceDetection_ieckai11.h5"]
    data_key_list = ['_0.0-0.0-0.0', '_0.0-0.0-0.2', '_0.0-0.0-0.4', '_0.0-0.0-0.6', '_0.0-0.0-0.8', '_0.0-0.0-1.0',
                     '_0.0-0.0-1.2', '_0.0-0.0-1.4', '_0.0-0.0-1.6', '_0.0-0.0-1.8', '_0.0-0.0-2.0', '_0.0-0.0-2.2',
                     '_0.0-0.0-2.4', '_0.0-0.0-2.6', '_0.0-0.0-2.8', '_0.0-0.0-3.0', '_0.0-0.0-3.2', '_0.0-0.0-3.4',
                     '_0.0-0.0-3.6', '_0.0-0.0-3.8', '_0.0-0.0-4.0', '_0.0-0.2-0.0', '_0.0-0.4-0.0', '_0.0-0.6-0.0',
                     '_0.0-0.8-0.0', '_0.0-1.0-0.0', '_0.0-1.2-0.0', '_0.0-1.4-0.0', '_0.0-1.6-0.0', '_0.0-1.8-0.0',
                     '_0.0-2.0-0.0', '_0.0-2.2-0.0', '_0.0-2.4-0.0', '_0.0-2.6-0.0', '_0.0-2.8-0.0', '_0.0-3.0-0.0',
                     '_0.0-3.2-0.0', '_0.0-3.4-0.0', '_0.0-3.6-0.0', '_0.0-3.8-0.0', '_0.0-4.0-0.0', '_0.2-0.0-0.0',
                     '_0.4-0.0-0.0', '_0.6-0.0-0.0', '_0.8-0.0-0.0', '_1.0-0.0-0.0', '_1.2-0.0-0.0', '_1.4-0.0-0.0',
                     '_1.6-0.0-0.0', '_1.8-0.0-0.0', '_2.0-0.0-0.0', '_2.2-0.0-0.0', '_2.4-0.0-0.0', '_2.6-0.0-0.0',
                     '_2.8-0.0-0.0', '_3.0-0.0-0.0', '_3.2-0.0-0.0', '_3.4-0.0-0.0', '_3.6-0.0-0.0', '_3.8-0.0-0.0',
                     '_4.0-0.0-0.0']

    # run train or test
    run(data_file_list, data_key_list, args)

    log_msg = 'Running time: ' + str((time.time() - start_time) / 60.0) + 'minutes'
    logger.info(log_msg)
