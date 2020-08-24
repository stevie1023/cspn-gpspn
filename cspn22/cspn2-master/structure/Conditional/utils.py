"""
Created on November 24, 2018

@author: Alejandro Molina
"""

import numpy as np


def concatenate_yx(y, x):
    y = y.reshape(x.shape[0], -1)
    return np.concatenate((y, x), axis=1)


def get_YX(data, feature_size):
    """
    decouples from the data, the labels and the features

    returns labels, features where data[labels|features] and |features| = evidence_size
    :param data:
    :param evidence_size:
    :return:
    """
    assert data is not None
    assert feature_size is not None
    assert feature_size > 0
    assert data.shape[1] > 1
    assert feature_size < data.shape[1], "feature size %s, data shape %s" % (
        feature_size,
        data.shape[1],
    )

    ncols = data.shape[1]
    y = data[:, 0 : ncols - feature_size].reshape(data.shape[0], -1)
    x = data[:, -feature_size:].reshape(-1, feature_size)

    assert y.shape[0] == x.shape[0] == data.shape[0]

    assert x.shape[1] == feature_size
    assert y.shape[1] + x.shape[1] == data.shape[1]
    return y, x


def get_Y(data, feature_size):
    """
    decouples from the data, the labels and the features

    returns labels where data[labels|features] and |features| = evidence_size
    :param data:
    :param evidence_size:
    :return:
    """
    y, _ = get_YX(data, feature_size)
    return y


def get_X(data, feature_size):
    """
    decouples from the data, the labels and the features

    returns features where data[labels|features] and |features| = evidence_size
    :param data:
    :param evidence_size:
    :return:
    """
    _, x = get_YX(data, feature_size)
    return x
