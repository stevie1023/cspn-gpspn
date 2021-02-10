import gc
import sys

import dill
import gpytorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prod_gp import ExactGPModel,GP,Sum,Productt
from prod_learnspngp import *
from gpytorch.kernels import *
from gpytorch.likelihoods import *
from torch.optim import *
from gpytorch.mlls import *
import torch
from scipy.io import arff
import random
sys.path.append("/home/mzhu/madesi/mzhu_code/")

#
random.seed(123)
np.random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed(123)

d_input = 8
# dataarff = arff.loadarff('/home/mzhu/madesi/mzhu_code/scm20d.arff')
# data = pd.DataFrame(dataarff[0])
data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14.csv')
# data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
# # data2 = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/100006dabnormal.csv')
data = pd.DataFrame(data).dropna()
train = data.sample(frac=0.8, random_state=58)
test = data.drop(train.index)
# data_ab = pd.DataFrame(data_ab).dropna()
# df = pd.read_csv('/home/mzhu/madesi/mzhu_code/VAR.csv',header=None)
# df = pd.DataFrame(df).dropna()  # miss = data.isnull().sum()/len(data)
# data2 = df.T
x_, y_ = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
y_d = y_.shape[1]
x1_, y1_ = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values
print(x_.shape)
std1, mu1= np.std(x_,axis=0), np.mean(y_,axis=0)
x = x_/ std1  # normalized train_x
x1 = x1_/std1 # test_x
y = y_-mu1# train_y
y1 = y1_-mu1 #test_y
# x_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_train.csv')
# x1_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_test.csv')
# y_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_train.csv')
# y1_= pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_test.csv')
#
# print(x_.shape)
# print(x1_.shape)
# mu1,std1 =x_.mean().to_numpy(),x_.std().to_numpy()
# # mu2,std2 = x1.mean(),x1.std()
# mu3,std3 =y_.mean().to_numpy(),y_.std().to_numpy()
# # mu4,std4 = y1.mean(),y1.std()
#
# x = x_/std1# normalized train_x
# x1 = x1_/std1 # test_x
# y = y_-mu3# train_y
# y1 = y1_-mu3 #test_y
# x = x.iloc[:,:].values
# x1 = x1.iloc[:,:].values
# y = y.iloc[:,:].values
# y1 = y1.iloc[:,:].values
# y_d = y.shape[1]
# d_input = x.shape[1]
MAEE=[]
RMSEE=[]
NLPDD=[]

for kkk in range(10):

    opts = {
        'min_samples': 0,
        'X': x,
        'Y': y,
        'max_depth': 100,
        'max_samples': 1100,
        'log': True,
        'jump': True,
        'reduce_branching': True
    }
    def _cached_gp(cache, **kwargs):
        min_, max_, y = list(kwargs['mins']), list(kwargs['maxs']), kwargs['y']
        cached = dict.get(cache, (*min_, *max_))
        if not cached:
            cache[(*min_, *max_)] = GPMixture(**kwargs)

        return cache[(*min_, *max_)]
    def build_bins(**kwargs):
        X = kwargs['X']
        Y = kwargs['Y']
        log = dict.get(kwargs, 'log', False)
        min_idx = dict.get(kwargs, 'max_samples', 1)

        root_mixture_opts = {
            'mins': np.min(X, 0),
            'maxs': np.max(X, 0),
            'n': len(X),
            'scope': [i for i in range(Y.shape[1])],
            'parent': None,
            'dimension': np.argsort(-np.var(X, axis=0))[0],
            'idx': X,
            'y':Y
        }

        root_node = Mixture(**root_mixture_opts)
        to_process, cache = [root_node], dict()
        # the size of leaves is around min_dex/2
        while len(to_process):
            node = to_process.pop()
            if type(node) is Product and type(node.children[0]) is Mixture:
                for i,child in enumerate(node.children):
                    d = child.dimension
                    x_node = child.idx
                    n_idx = child.n
                    mins_loop, maxs_loop = np.min(x_node, 0), np.max(x_node, 0)
                    scope = child.scope
                    next_depth = child.depth + 1
                    d_selected = np.argsort(-np.var(x_node, axis=0))
                    d2 = d_selected[1]
                    d3 = d_selected[2]
                    node_splits_all = [1, 2]
                    d = [d, d2, d3]

                    if len(scope) < 3:
                        gp = []
                        prod_opts = {
                            'minsy': mins_loop,
                            'maxsy': maxs_loop,
                            'scope': scope,
                            'children': gp,
                        }
                        prod = Product(**prod_opts)
                        results=[]
                        for yi in prod.scope:
                            a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=n_idx, y=yi, parent=None)
                            gp.append(a)
                        results.append(prod)
                        to_process.extend(results)  # results are put into the root_reigon
                        child.children.extend(results)

                    else:
                        n=0
                        for split in node_splits_all:
                            # y_idx = y[idx]
                            next_dimension = d[n]
                            results = []

                            a = int(len(scope) / 2)
                            scope1 = random.sample(scope, a)
                            scope2 = list(set(scope) - set(scope1))
                            mixture_opts1 = {
                                'mins': mins_loop,
                                'maxs': maxs_loop,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': n_idx,
                                'scope': scope1,
                                'idx':x_node
                            }
                            mixture_opts2 = {
                                'mins': mins_loop,
                                'maxs': maxs_loop,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': n_idx,
                                'scope': scope2,
                                'idx':x_node
                            }
                            prod_opts = {
                                'minsy': mins_loop,
                                'maxsy': maxs_loop,
                                'scope': scope1 + scope2,
                                'children': [Mixture(**mixture_opts1), Mixture(**mixture_opts2)]
                            }

                            prod = Product(**prod_opts)
                            results.append(prod)
                            n += 1

                            if len(results) > 1:
                                to_process.extend(results)  # results are put into the root_reigon
                                child.children.append(results)  # create product nodes for every mixture node
                            elif len(results) == 1:
                                to_process.extend(results)  # results are put into the root_reigon
                                child.children.extend(results)
                            else:
                                raise Exception('1')

            elif type(node) is Mixture:

                d = node.dimension
                x_node = node.idx
                n_idx = node.n
                mins_loop, maxs_loop = np.min(x_node, 0), np.max(x_node, 0)
                scope = node.scope
                next_depth = node.depth + 1
                d_selected = np.argsort(-np.var(x_node, axis=0))
                d2 = d_selected[1]
                d3 = d_selected[2]
                node_splits_all = [1, 2]
                d = [d, d2, d3]
                m=0

                if len(scope) < 3:
                    gp = []
                    prod_opts = {
                        'minsy': mins_loop,
                        'maxsy': maxs_loop,
                        'scope': scope,
                        'children': gp,
                    }
                    prod = Product(**prod_opts)
                    results=[]
                    for yi in prod.scope:
                        a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=n_idx, y=yi, parent=None)
                        gp.append(a)
                    results.append(prod)

                    to_process.extend(results)  # results are put into the root_reigon
                    node.children.extend(results)

                else:
                    for split in node_splits_all:
                        # y_idx = y[idx]
                        next_dimension = d[m]
                        results=[]

                        a = int(len(scope) / 2)
                        scope1 = random.sample(scope, a)
                        scope2 = list(set(scope) - set(scope1))
                        mixture_opts1 = {
                            'mins': mins_loop,
                            'maxs': maxs_loop,
                            'depth': next_depth,
                            'dimension': next_dimension,
                            'n': n_idx,
                            'scope': scope1,
                            'idx': x_node
                        }
                        mixture_opts2 = {
                            'mins': mins_loop,
                            'maxs': maxs_loop,
                            'depth': next_depth,
                            'dimension': next_dimension,
                            'n': n_idx,
                            'scope': scope2,
                            'idx':x_node
                        }
                        prod_opts = {
                            'minsy': mins_loop,
                            'maxsy': maxs_loop,
                            'scope': scope1 + scope2,
                            'children': [Mixture(**mixture_opts1), Mixture(**mixture_opts2)]
                        }

                        prod = Product(**prod_opts)
                        results.append(prod)
                        m+=1
                        if len(results) >1 :
                            to_process.extend(results)  # results are put into the root_reigon
                            node.children.append(results)  # create product nodes for every mixture node
                        elif len(results) ==1:
                            to_process.extend(results)  # results are put into the root_reigon
                            node.children.extend(results)
                        else:
                            raise Exception('1')


        gps = list(cache.values())

        aaa = [gp.idx for gp in gps]
        c = (np.mean(aaa) ** 3) * len(aaa)

        r = 1 - (c / (len(X) ** 3))

        print("Full:\t\t", len(X) ** 3, "\nOptimized:\t", int(c),
              f"\n#GP's:\t\t {len(gps)} ({np.min(aaa)}-{np.max(aaa)})",

              "\nReduction:\t", f"{round(100 * r, 4)}%")

        print(f"Lengths:\t {aaa}\nSum:\t\t {sum(aaa)} (N={len(X)})")

        return root_node, gps



    def structure2(root_region, scope,**kwargs):
        count=0
        root = Sum(scope=scope)
        to_process, gps = [(root_region, root)], dict()

        while len(to_process):
            gro, sto = to_process.pop()
            if type(gro) is Mixture:
                for child in gro.children:
                    if type(child) is Product:
                        scope = child.scope
                        _child = Productt(scope = scope)
                        sto.children.append(_child)
                        _cn = len(sto.children)
                        sto.weights = np.ones(_cn) / _cn
                    else:
                        print(type(child))
                        raise Exception('1')
                to_process.extend(zip(gro.children, sto.children))
            elif type(gro) is Product:
                i = 0
                for child in gro.children:
                    if type(child) is Mixture:
                        scope = child.scope
                        _child = Sum(scope = scope)
                        sto.children.append(_child)
                    elif type(child) is GPMixture:
                        gp_type = 'matern1.5_ard'
                        scopee = gro.scope
                        # key = (*gro.mins, *gro.maxs, gp_type,gro.collect,count,scopee[i])
                        # gps[key] = GP(type=gp_type, mins=gro.mins, maxs=gro.maxs,collect = gro.collect,count = count, scope=scopee[i])
                        key = (*gro.mins, *gro.maxs, gp_type, count, scopee[i])
                        gps[key] = GP(type=gp_type, mins=gro.mins, maxs=gro.maxs, count=count,
                                      scope=scopee[i])
                        count+=1
                        sto.children.append(gps[key])
                        i += 1
                    else:
                        raise Exception('1')

                to_process.extend(zip(gro.children, sto.children))
        return root, list(gps.values())


    root_region, gps_ = build_bins(**opts)
    root, gps = structure2(root_region,scope = [i for i in range(y.shape[1])], gp_types=['matern1.5_ard'])


    lr = 0.1
    steps = 300
    loss = []
    for i, gp in enumerate(gps):
        idx = query(x, gp.mins, gp.maxs)
        gp.x = x[idx]
        y_scope = y[:,gp.scope]
        gp.y = y_scope[idx]
        print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
        print('scope', gp.scope)
        gp.init3(cuda=True,lr = 0.2,steps=200)

    root.update()

    mae = 0
    rmse = 0
    # np.savetxt('loss_train_sumGP_200.csv', [loss_train], delimiter=',')

    mu, cov = root.forward(x1[:, :], smudge=0,y_d = y_d)

    for k in range(y.shape[1]):
        mu_s1 = mu[:, 0, k]
        sqe1 = (mu_s1 - y1[:, k]) ** 2
        rmse1 = np.sqrt(sqe1.sum() / len(y1))
        mae1 = np.sqrt(sqe1).sum() / len(y1)
        mae += mae1
        rmse += rmse1
    nlpd1 = 0
    count = 0
    for i in range(mu.shape[0]):
        sigma = np.sqrt(np.abs(np.linalg.det(cov[i, :, :])))
        if sigma == 0:
            count += 1
            continue
        # d1 = (test.iloc[i,d_input:].values.reshape((1,1,y_d))-mu[i,:,:]).reshape((1,y_d))
        d1 = (y1[i, :].reshape((1, 1, y_d)) - mu[i, :, :]).reshape((1, y_d))

        a = 1 / (np.power((2 * np.pi), y.shape[1] / 2) * sigma)
        ni = np.linalg.pinv(cov[i, :, :])
        b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
        if b > 0.00000000000001:
            nlpd = -np.log(b)
        else:
            nlpd = 0

        nlpd1 += nlpd
    nlpd2 = nlpd1 / len(y1)

    RMSEE.append(rmse / y_d)
    MAEE.append(mae / y_d)
    NLPDD.append(nlpd2)

# #
print(f"SPN-GP  RMSE: {RMSEE}")
print(f"SPN-GP  MAE1: {MAEE}")
print(f"SPN-GP  NLPD1: {NLPDD}")
print(count)
print(f"SPN-GP  RMSE mean: {np.mean(np.array(RMSEE))} std:{np.std(np.array(RMSEE))}")
print(f"SPN-GP  MAE mean: {np.mean(np.array(MAEE))} std:{np.std(np.array(MAEE))}")
print(f"SPN-GP  NLPD mean: {np.mean(np.array(NLPDD))} std:{np.std(np.array(NLPDD))}")


