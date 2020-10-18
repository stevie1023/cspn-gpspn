import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from learnspngp import build, query, build_bins
from spngp import structure
import sys


y_d = 2
np.random.seed(58)
data = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/10000normal.csv')
data2 = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/100006dabnormal.csv')
data = pd.DataFrame(data).dropna() # miss = data.isnull().sum()/len(data)
dmean, dstd = data.mean(), data.std()
data = (data-dmean)/dstd
data2 = pd.DataFrame(data2).dropna() # miss = data.isnull().sum()/len(data)
dmean2, dstd2 = data2.mean(), data2.std()
data2 = (data2-dmean2)/dstd2
# GPSPN on full data
train = data.sample(frac=0.8, random_state=58)
test  = data.drop(train.index)
x, y = train.iloc[:, 3:].values, train.iloc[:, y_d].values.reshape(-1,1)

train2 = data2.sample(frac=0.8, random_state=58)
test2  = data2.drop(train2.index)

opts = {
    'min_samples':          0,
    'X':                    x,
    'qd':                   4,
    'max_depth':            10,
    'max_samples':       3000,
    'log':               True,
    'jump':              True,
    'reduce_branching':  True
}
root_region, gps_ = build_bins(**opts)

root, gps = structure(root_region, gp_types=['rbf'])  #modified



for i, gp in enumerate(gps):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]

    print(f"Training GP set1 {i+1}/{len(gps)} ({len(idx)})") #modified
    gp.init(cuda=True)

##modified
# for i, gp in enumerate(gps1):
#     idx = query(x, gp.mins, gp.maxs)
#     gp.x, gp.y = x[idx], y[idx]
#
#     print(f"Training GP set2 {i+1}/{len(gps)} ({len(idx)})") #modified
#     gp.init1(cuda=True)
#
# for i, gp in enumerate(gps2):
#     idx = query(x, gp.mins, gp.maxs)
#     gp.x, gp.y = x[idx], y[idx]
#
#     print(f"Training GP set3 {i+1}/{len(gps)} ({len(idx)})") #modified
#     gp.init2(cuda=True)

mu, cov = root.forward(test.iloc[:, 3:].values,smudge=0)
mu_s1 = (mu[:,0].ravel() * dstd.iloc[y_d]) + dmean.iloc[y_d]

mu_t1 = (test.iloc[:, y_d] * dstd.iloc[y_d]) + dmean.iloc[y_d]

sqe1 = (mu_s1 - mu_t1.values) ** 2

rmse1 = np.sqrt(sqe1.sum() / len(test))

mae1 = np.sqrt(sqe1).sum() / len(test)
nlpd1=0
for i in range(mu.shape[0]):
    sigma2 =np.abs(cov[i,:])
    d1 = (test.iloc[i, y_d]-mu[i,:])
    a = np.sqrt(1/((2*np.pi)*sigma2))
    b=a * np.exp(-0.5 * np.power(d1,2)/sigma2)
    if b>0.0000000001:
        nlpd = -np.log(a * np.exp(-0.5 * np.power(d1,2)/sigma2))
    else:
        nlpd = 0
    nlpd1+=nlpd

nlpd2 = nlpd1/len(test)


print(f"SPN-GP  RMSE1: {rmse1}")
print(f"SPN-GP  MAE1: {mae1}")
print(f"SPN-GP  NLPD1: {nlpd2}")
#

