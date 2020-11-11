
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prod_structure import query, build_bins
from prod_inference import structure
import sys
from sklearn.linear_model import SGDRegressor
import pickle
import dill
from sklearn.linear_model import BayesianRidge, LinearRegression
from scipy import stats

x= pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_train.csv')
x1 = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/x_test.csv')
y = pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_train.csv')
y1= pd.read_csv('/home/mzhu/madesi/datasets/datasets/parkinsons/y_test.csv')
#
mu1,std1 =x.mean(),x.std()
mu2,std2 = x1.mean(),x1.std()
mu3,std3 =y.mean(),y.std()
mu4,std4 = y1.mean(),y1.std()
x = (x-mu1)/std1 # normalized train_x
x1 = (x1-mu2)/std2 # test_x
y = (y-mu3)/std3 # train_y
y1 = (y1-mu4)/std4 #test_y
y_d = y.shape[1]
x = x.iloc[:,:].values
x_original = x.copy()
x1 = x1.iloc[:,:].values
y = y.iloc[:,:].values
y_orginal = y.copy()
y1 = y1.iloc[:,:].values
noise = np.random.normal(0, .01, x.shape)
x = noise + x
noise2 = np.random.normal(0, .01, y.shape)
y = noise2 + y
opts = {
    'min_samples': 0,
    'X': x,
    'Y': y,
    'qd': 4,
    'max_depth': 4,
    'max_samples': 10 ** 10,
    'log': True,
    'jump': True,
    'reduce_branching': True
}
root_region, gps_ = build_bins(**opts)

root, gps = structure(root_region,scope = [i for i in range(y.shape[1])], gp_types=['rbf'])

for i, gp in enumerate(gps):
    idx = query(x, gp.mins, gp.maxs)
    gp.x = x_original[idx]
    y_scope = y_orginal[:,gp.scope]

    gp.y = y_scope[idx]

    print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
    gp.init(cuda=True)

root.update()
mll = root.mll

# filename = 'usflight4childrennew.dill'
# dill.dump(root, open(filename, 'wb'))

mu, cov= root.forward(x1[:,:], smudge=0)

mu_s1 = mu[:,0, 0]
mu_s2 = mu[:,0, 1]

sqe1 = (mu_s1 - y1[:,0]) ** 2
sqe2 = (mu_s2 - y1[:,1]) ** 2
#
rmse1 = np.sqrt(sqe1.sum() / len(y1))
rmse2 = np.sqrt(sqe2.sum() / len(y1))

mae1 = np.sqrt(sqe1).sum() / len(y1)
mae2 = np.sqrt(sqe2).sum() / len(y1)


nlpd1=0
# nlpd for multivariate gaussian distribution

for i in range(mu.shape[0]):
    sigma = np.sqrt(np.abs(np.linalg.det(cov[i,:,:])))
    # d1 = (test.iloc[i,:3].values.reshape((1,1,y_d))-mu[i,:,:]).reshape((1,y_d))
    d1 = (y1[i, :].reshape((1, 1, y_d)) - mu[i, :, :]).reshape((1, y_d))
    a = 1/(np.power((2*np.pi),y.shape[1]/2)*sigma)
    ni =np.linalg.pinv(cov[i, :, :])
    b = a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T))
    if b > 0.0000000001:
        nlpd = -np.log(b)
    else:
        nlpd = 0

    nlpd1+=nlpd

nlpd2 = nlpd1/len(y1)

print(f"SPN-GP  RMSE1: {rmse1}, RMSE2: {rmse2}")
print(f"SPN-GP  MAE1: {mae1}, MAE2: {mae2}")
print(f"SPN-GP  NLPD: {nlpd2}")

