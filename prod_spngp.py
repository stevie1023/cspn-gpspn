import gc
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpytorch.priors import GammaPrior
from prod_learnspngp import query, build_bins
from prod_gp import structure, ExactGPModel
from gpytorch.kernels import *
from gpytorch.likelihoods import *
from torch import optim
from torch.optim import *
from gpytorch.mlls import *
import torch
from scipy.io import arff

import gpytorch
# from pytorchtools import EarlyStopping
import sys
from sklearn.linear_model import SGDRegressor
import pickle
import dill
from sklearn.linear_model import BayesianRidge, LinearRegression
from scipy import stats
# d_input = 8
np.random.seed(58)
# # dataarff = arff.loadarff('/home/mzhu/madesi/mzhu_code/scm20d.arff')
# # data = pd.DataFrame(dataarff[0])
# data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14.csv')
# # data = pd.read_csv('/home/mzhu/madesi/mzhu_code/windmill14_ab.csv')
# # # data2 = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/100006dabnormal.csv')
# data2 = pd.DataFrame(data).dropna()
# # data_ab = pd.DataFrame(data_ab).dropna()
# # df = pd.read_csv('/home/mzhu/madesi/mzhu_code/VAR.csv',header=None)
# # df = pd.DataFrame(df).dropna()  # miss = data.isnull().sum()/len(data)
# # data2 = df.T
# dmean, dstd = data2.mean(), data2.std()
# data = (data2 - dmean) / dstd
# # dmean_, dstd_ = data_ab.mean(), data_ab.std()
# # data_ab = (data_ab - dmean_) / dstd_
# # train_ab = data_ab.sample(frac=0.8, random_state=58)
# # test_ab = data_ab.drop(train_ab.index)
# # x2, y2 = test_ab.iloc[:, :d_input].values, test_ab.iloc[:, d_input:].values
#
# train = data.sample(frac=0.8, random_state=58)
# test = data.drop(train.index)
# x, y = train.iloc[:, :d_input].values, train.iloc[:, d_input:].values
# y_d = y.shape[1]
# x1, y1 = test.iloc[:, :d_input].values, test.iloc[:, d_input:].values
# print(x.shape)


# # X1 = np.sort(np.random.rand(800))[:, None]
# # X2 = np.sort(np.random.rand(800))[:, None]
# # X3 = np.sort(np.random.rand(800))[:, None]
# # X4 = np.sort(np.random.rand(800))[:, None]
# # noise = np.random.normal(0,.1,(2000,1))
# # X = [X1,X2,X3,X4]
# #
# #
# # def experiment_true_u_functions(X_list):
# #     u_functions = []
# #     for X in X_list:
# #         u_task = np.empty((X.shape[0], 1))
# #         u_task[:, 0, None] = (4.5 * np.cos(2 * np.pi * X + 1.5 * np.pi) - \
# #                              3 * np.sin(4.3 * np.pi * X + 0.3 * np.pi) + \
# #                              5 * np.cos(7 * np.pi * X + 2.4 * np.pi))
# #         # u_task[:, 1, None] = (4.5 * np.cos(1.5 * np.pi * X + 0.5 * np.pi) + \
# #         #                      5 * np.sin(3 * np.pi * X + 1.5 * np.pi) - \
# #         #                      5.5 * np.cos(8 * np.pi * X + 0.25 * np.pi))
# #         u_functions.append(u_task)
# #     return u_functions
# #
# # # True functions values for inputs X
# # Y = np.concatenate(experiment_true_u_functions(X),axis=1)
# # X = np.concatenate(X,axis=1)
# # print(Y)
# # index = np.random.choice(800,600)
# # y_d = 4
# # x = X[index,:]
# # y = Y[index,:]
# # x1 = np.delete(X,[index],0)
# # y1 = np.delete(Y,[index],0)
# #
# # mu1,std1 =x.mean(),x.std()
# # mu2,std2 = x1.mean(),x1.std()
# # mu3,std3 =y.mean(),y.std()
# # mu4,std4 = y1.mean(),y1.std()
# # x = (x-mu1)/std1 # normalized train_x
# # x1 = (x1-mu2)/std2 # test_x
# # y = (y-mu3)/std3 # train_y
# # y1 = (y1-mu4)/std4 #test_y
# # print(y1)
# #
x_= pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_train.csv')
x1_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_test.csv')
y_ = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_train.csv')
y1_= pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_test.csv')

print(x_.shape)
print(x1_.shape)
mu1,std1 =x_.mean().to_numpy(),x_.std().to_numpy()
# mu2,std2 = x1.mean(),x1.std()
mu3,std3 =y_.mean().to_numpy(),y_.std().to_numpy()
# mu4,std4 = y1.mean(),y1.std()

x = x_/std1# normalized train_x
x1 = x1_/std1 # test_x
y = y_-mu3# train_y
y1 = y1_-mu3 #test_y
# x = (x-x.min())/(x.max()-x.min())
# x1 = (x1-x1.min())/(x1.max()-x1.min())
# y = (y-y.min())/(y.max()-y.min())
# y1 = (y1-y1.min())/(y1.max()-y1.min())
x = x.iloc[:,:].values
x1 = x1.iloc[:,:].values
y = y.iloc[:,:].values
y1 = y1.iloc[:,:].values
y_d = y.shape[1]
d_input = x.shape[1]
# #
opts = {
    'min_samples': 0,
    'X': x,
    'Y': y,
    'qd': 1,
    'max_depth': 100,
    'max_samples': 1010,
    'log': True,
    'jump': True,
    'reduce_branching': True
}
root_region, gps_ = build_bins(**opts)

root, gps = structure(root_region,scope = [i for i in range(y.shape[1])], gp_types=['rbf_ard'])


lr = 0.1
steps = 150
noises = torch.ones(100) * 0.001

# self.likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
likelihood_scope = [GaussianLikelihood().train() for _ in range(y_d)]
tensor_x = torch.from_numpy(np.zeros((100,d_input))).float().to('cuda')
tensor_y = torch.from_numpy(np.zeros((100,d_input))).float().to('cuda')
model_scope = [ExactGPModel(x = tensor_x,y = tensor_y,likelihood = likelihood_scope[i], type='matern1.5_ard') for i in range(y.shape[1])]
# opt = [model_scope[p].parameters() for p in range(y_d)]
# optimizer_scope = [Adam([{'params':opt[p]}], lr=lr) for p in range(y_d)]
l0=[]
for m in range(y_d):
    l0.extend(list(model_scope[m].parameters()))
optimizer_scope = Adam([{'params':l0}], lr=lr)
# optimizer_scope=optim.SGD(l0, lr=0.1, momentum=0.9)
# optimizer_scope = torch.optim.LBFGS(l0)
model_scope = [i.to('cuda') for i in model_scope]

nlpd = []
loss_train = []
for i in range(steps): #这是优化的大循环，优化共#steps步
    # tree_loss = [0] * y.shape[1]
    # tree_scope = [0] * y.shape[1]
    optimizer_scope.zero_grad()
    for j, gp in enumerate(gps):
        if i == 0:
            idx = query(x, gp.mins, gp.maxs)
            gp.x = x[idx]
            y_scope = y[:,gp.scope]
            gp.y = y_scope[idx]
            gp.n = len(gp.x)
        cuda_ = True
        temp_device = torch.device("cuda" if cuda_ else "cpu")
        if cuda_:
            torch.cuda.empty_cache()
        x_temp = torch.from_numpy(gp.x).float().to(temp_device)
        y_temp = torch.from_numpy(gp.y.ravel()).float().to(temp_device)
        model_scope[gp.scope].set_train_data(inputs=x_temp, targets=y_temp, strict=False)
        model_scope[gp.scope].train()
        gp.likelihood = likelihood_scope[gp.scope]
        gp.model = model_scope[gp.scope]
        mll = ExactMarginalLogLikelihood(likelihood_scope[gp.scope], model_scope[gp.scope])

        output = model_scope[gp.scope](x_temp)  # Output from model
        if i == steps-1:
            gp.mll = mll(output, y_temp).item()
        gp.mll_grad = -mll(output, y_temp)
        # loss = -mll(output, y_temp)
        x_temp.detach()
        y_temp.detach()
        del x_temp
        del y_temp
        del gp.model,gp.likelihood
        x_temp = y_temp = None
        torch.cuda.empty_cache()
        gc.collect()
        # loss_scope[gp.scope]+=gp.mll_grad

    tree_loss_all = root.update_mll()
    loss_train.append(tree_loss_all.item())
    print('loss', tree_loss_all.item())
    tree_loss_all.backward()
    optimizer_scope.step()

# root.update()

for i, gp in enumerate(gps):
    x_temp = torch.from_numpy(gp.x).float().to('cuda')
    y_temp = torch.from_numpy(gp.y.ravel()).float().to('cuda')
    model_scope[gp.scope].set_train_data(inputs=x_temp, targets=y_temp, strict=False)
    gp.model = model_scope[gp.scope]
    gp.likelihood = likelihood_scope[gp.scope]
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp.model)
    output = model_scope[gp.scope](x_temp)
    gp.mll = mll(output, y_temp).item()
    x_temp.detach()
    y_temp.detach()
    del x_temp
    del y_temp
    x_temp = y_temp = None
    torch.cuda.empty_cache()
root.update()
np.savetxt('park300.csv', [loss_train], delimiter=',')
# #
# for i, gp in enumerate(gps):
#     idx = query(x, gp.mins, gp.maxs)
#     gp.x = x[idx]
#     y_scope = y[:,gp.scope]
#     gp.y = y_scope[idx]
#     print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
#     gp.init(cuda=True)
# root.update()
#
# filename = 'windmill200.dill'
# dill.dump(root, open(filename, 'wb'))

# filename = 'var200_0.1_spgpn.dill'
# dill.dump(root, open(filename, 'wb'))
mu, cov= root.forward(x1[:,:], smudge=0,y_d = y_d)

# # mu2, cov2 = root.forward(test2.iloc[:, 3:].values, smudge=0)
#
rmse = 0
mae = 0
for k in range(y.shape[1]):
    mu_s1 = mu[:,0, k]
    sqe1 = (mu_s1 - y1[:,k]) ** 2
    rmse1 = np.sqrt(sqe1.sum() / len(y1))
    mae1 = np.sqrt(sqe1).sum() / len(y1)
    mae+=mae1
    rmse+=rmse1
    # np.savetxt('rmse_windmill.csv', [all_rmse_improved], delimiter=',')

nlpd1=0
count=0
for i in range(mu.shape[0]):
    sigma = np.sqrt(np.abs(np.linalg.det(cov[i,:,:])))
    if sigma == 0:
        count+=1
        continue
    # d1 = (test.iloc[i,d_input:].values.reshape((1,1,y_d))-mu[i,:,:]).reshape((1,y_d))
    d1 = (y1[i, :].reshape((1, 1, y_d)) - mu[i, :, :]).reshape((1, y_d))
    a = 1/(np.power((2*np.pi),y.shape[1]/2)*sigma)
    ni =np.linalg.pinv(cov[i, :, :])
    b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
    if b > 0.0000000001:
        nlpd = -np.log(b)
    else:
        nlpd = 0

    nlpd1+=nlpd
#
nlpd2 = nlpd1/len(y1)
# nlpd2 = nlpd1/len(test)
# #
print(f"SPN-GP  RMSE: {rmse/y_d}")
print(f"SPN-GP  MAE: {mae/y_d}")
print(f"SPN-GP  NLPD: {nlpd2}")
print(count)

