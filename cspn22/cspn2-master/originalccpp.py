

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from originallearnspngp import build, query, build_bins
from originalspngp import structure
import sys
from sklearn.linear_model import SGDRegressor

from sklearn.linear_model import BayesianRidge, LinearRegression
from scipy import stats
# np.random.seed(58)
data = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/ccpp.csv')
# data = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/30000normal.csv')
data = pd.DataFrame(data).dropna()  # miss = data.isnull().sum()/len(data)
dmean, dstd = data.mean(), data.std()
data = (data - dmean) / dstd

train = data.sample(frac=0.8, random_state=58)
test = data.drop(train.index)
x, y = train.iloc[:, 2:].values, train.iloc[:, 0].values.reshape(-1, 1)

# x= pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/x_train.csv')
# x1 = pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/x_train.csv')
# y = pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/x_train.csv')
# y1= pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/x_train.csv')
#
# mu1,std1 =x.mean(),x.std()
# mu2,std2 = x1.mean(),x1.std()
# mu3,std3 =y.mean(),y.std()
# mu4,std4 = y1.mean(),y1.std()
# x = (x-mu1)/std1
# x1 = (x1-mu2)/std2
# y = (y-mu3)/std3
# y1 = (y1-mu4)/std4
#
# x = x.iloc[4000:8000,:].values
# x1 = x1.iloc[200:,].values
# y = y.iloc[4000:8000,0].values
# y1 = y1.iloc[:200,0].values
#
# SNR=5
# noise = np.random.randn(x.shape[0],x.shape[1]) 	#产生N(0,1)噪声数据
# noise = noise-np.mean(noise) 								#均值为0
# signal_power = np.linalg.norm( x )**2 / x.size	#此处是信号的std**2
# noise_variance = signal_power/np.power(10,(SNR/10))         #此处是噪声的std**2
# noise = (np.sqrt(noise_variance) / np.std(noise) )*noise    ##此处是噪声的std**2
# x = noise + x


# sgdregressor on full data
# clf = SGDRegressor()
# clf.fit(x, y)
# y_pred = clf.predict(test.iloc[:, 0:4])
# y_pred = y_pred *dstd.iloc[-1] + dmean.iloc[-1]
# mu_t = (test.iloc[:, -1] * dstd.iloc[-1]) + dmean.iloc[-1]
# sqe = (y_pred - mu_t.values) ** 2
#
# rmse = np.sqrt(sqe.sum() / len(test))
# print(rmse)

# bayesian ridge regression
# lambda_ = 4.
# n_features = x.shape[1]
# w = np.zeros(n_features)
# relevant_features = np.random.randint(0, n_features, 10)
# for i in relevant_features:
#     w[i] = stats.norm.rvs(loc=0, scale=1. / np.sqrt(lambda_))
# # Create noise with a precision alpha of 50.
# alpha_ = 1.
# noise = stats.norm.rvs(loc=0, scale=1. / np.sqrt(alpha_), size=x.shape[0])
# # Create the target
# y = np.dot(x, w) + noise
# # Fit the Bayesian Ridge Regression and an OLS for comparison
# clf = BayesianRidge(compute_score=True)
# clf.fit(x, y)

# Plotting some predictions for polynomial regression
# def f(x, noise_amount):
#     y = np.sqrt(x) * np.sin(x)
#     noise = np.random.normal(0, 1, len(x))
#     return y + noise_amount * noise
#
# degree = 10
# X = np.linspace(0, 10, 100)
# y = f(X, noise_amount=0.1)
# clf_poly = BayesianRidge()
# clf_poly.fit(np.vander(X, degree), y)
#
# X_plot = np.linspace(0, 11, 25)
# y_plot = f(X_plot, noise_amount=0)
# y_mean, y_std = clf_poly.predict(np.vander(X_plot, degree), return_std=True)
# plt.figure(figsize=(6, 5))
# plt.errorbar(X_plot, y_mean, y_std, color='navy',
#              label="Polynomial Bayesian Ridge Regression", linewidth=lw)
# plt.plot(X_plot, y_plot, color='gold', linewidth=lw,
#          label="Ground Truth")
# plt.ylabel("Output y")
# plt.xlabel("Feature X")
# plt.legend(loc="lower left")
# plt.show()
# y_pred = clf.predict(test.iloc[:, 0:4])
# y_pred = y_pred *dstd.iloc[-1] + dmean.iloc[-1]
# mu_t = (test.iloc[:, -1] * dstd.iloc[-1]) + dmean.iloc[-1]
# sqe = (y_pred - mu_t.values) ** 2
#
# rmse = np.sqrt(sqe.sum() / len(test))
# print(rmse)

opts = {
    'min_samples': 0,
    'X': x,
    'qd': 4,
    'max_depth': 4,
    'max_samples': 10 ** 10,
    'log': True,
    'jump': True,
    'reduce_branching': True
}
root_region, gps_ = build_bins(**opts)
# root_region, gps_ = build(**opts)
# root_region, gps_ = build(X=x, delta_divisor=3, max_depth=2)
root, gps = structure(root_region, gp_types=['rbf'])

for i, gp in enumerate(gps):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]
    print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
    gp.init(cuda=True)

root.update()

#
mu_s, cov_s = root.forward(test.iloc[:, 2:].values, smudge=0)
print(mu_s)
mu_s = (mu_s.ravel() * dstd.iloc[0]) + dmean.iloc[0]
mu_t = (test.iloc[:, 0] * dstd.iloc[0]) + dmean.iloc[0]
sqe = (mu_s - mu_t.values) ** 2
rmse = np.sqrt(sqe.sum() / len(test))
mae = np.sqrt(sqe).sum() / len(test)
print(f"SPN-GP (smudge=0 RMSE: {rmse}")
print('mae',mae)

# mu_s, cov_s = root.forward(x1, smudge=0)
# mu_s = (mu_s.ravel() * std2.iloc[0]) + mu2.iloc[0]
# mu_t = (y1 * std4[0]) + mu4[0]
# sqe = (mu_s - mu_t) ** 2
# rmse = np.sqrt(sqe.sum() / len(y1))
# print(f"SPN-GP (smudge=0) \t RMSE: {rmse}")
# mae = np.sqrt(sqe).sum() / len(y1)
# print('mae',mae)