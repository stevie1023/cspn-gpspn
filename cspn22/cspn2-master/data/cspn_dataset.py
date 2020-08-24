from os.path import dirname

import numpy as np

path = dirname(__file__) + "/"

def get_binary_data(name):
    train = np.loadtxt(path + "binary/" + name + ".ts.data", dtype=float, delimiter=",", skiprows=0)
    test = np.loadtxt(path + "binary/" + name + ".test.data", dtype=float, delimiter=",", skiprows=0)
    valid = np.loadtxt(path + "binary/" + name + ".valid.data", dtype=float, delimiter=",", skiprows=0)
    D = np.vstack((train, test, valid))
    F = D.shape[1]
    features = ["V" + str(i) for i in range(F)]

    return (
    name.upper(), np.asarray(features), D, train, test, np.asarray(["discrete"] * F), np.asarray(["bernoulli"] * F))


def get_binary_mask(name, mask_type='ev50'):
    train = np.genfromtxt(path + "/" + mask_type + '/' + name + ".ts.ev", dtype=float, delimiter=",")
    test = np.genfromtxt(path + "/" + mask_type + '/' + name + ".test.ev", dtype=float, delimiter=",")
    valid = np.genfromtxt(path + "/" + mask_type + '/' + name + ".valid.ev", dtype=float, delimiter=",")
    D = np.vstack((train, test, valid))
    F = D.shape[1]
    features = ["V" + str(i) for i in range(F)]

    return (
    name.upper(), np.asarray(features), D, train, test, np.asarray(["discrete"] * F), np.asarray(["bernoulli"] * F))


