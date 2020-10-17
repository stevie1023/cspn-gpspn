
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prod_structure import query, build_bins
from prod_inference import structure
import sys
from sklearn.linear_model import SGDRegressor

from sklearn.linear_model import BayesianRidge, LinearRegression
from scipy import stats
np.random.seed(58)
data = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/10000normal.csv')
data = pd.DataFrame(data).dropna()  # miss = data.isnull().sum()/len(data)
dmean, dstd = data.mean(), data.std()
data = (data - dmean) / dstd

train = data.sample(frac=0.8, random_state=58)
test = data.drop(train.index)

x, y = train.iloc[:, 3:].values, train.iloc[:, :3].values


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
    gp.x = x[idx]
    y_scope = y[:,gp.scope]
    print(gp.scope)
    gp.y = y_scope[idx]

    print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
    gp.init(cuda=True)

root.update()

#
mu, cov = root.forward(test.iloc[:, 3:].values, smudge=0)
mu_s1 = (mu[:,0,0].ravel() * dstd.iloc[0]) + dmean.iloc[0]
mu_s2 = (mu[:,0,1].ravel() * dstd.iloc[1]) + dmean.iloc[1]
mu_s3 = (mu[:,0,2].ravel() * dstd.iloc[2]) + dmean.iloc[2]

mu_t1 = (test.iloc[:, 0] * dstd.iloc[0]) + dmean.iloc[0]
mu_t2 = (test.iloc[:, 1] * dstd.iloc[1]) + dmean.iloc[1]
mu_t3 = (test.iloc[:, 2] * dstd.iloc[2]) + dmean.iloc[2]

sqe1 = (mu_s1 - mu_t1.values) ** 2
sqe2 = (mu_s2 - mu_t2.values) ** 2
sqe3 = (mu_s3 - mu_t3.values) ** 2
rmse1 = np.sqrt(sqe1.sum() / len(test))
rmse2 = np.sqrt(sqe2.sum() / len(test))
rmse3 = np.sqrt(sqe3.sum() / len(test))
mae1 = np.sqrt(sqe1).sum() / len(test)
mae2 = np.sqrt(sqe2).sum() / len(test)
mae3 = np.sqrt(sqe3).sum() / len(test)

nlpd1=0
# nlpd for multivariate gaussian distribution
for i in range(mu.shape[0]):
    sigma = np.sqrt(np.abs(np.linalg.det(cov[i,:,:])))
    d1 = (test.iloc[i, :3].values.reshape((1,1,3))-mu[i,:,:]).reshape((1,3))
    a = 1/(np.power((2*np.pi),y.shape[1]/2)*sigma)
    ni =np.linalg.pinv(cov[i, :, :])
    nlpd = -np.log(a * np.exp(np.dot(np.dot(-1 / 2 * d1, ni), d1.T)))

    nlpd1+=nlpd

nlpd2 = nlpd1/len(test)


print(f"SPN-GP  RMSE1: {rmse1}, RMSE2: {rmse2},RMSE3: {rmse3}")
print(f"SPN-GP  MAE1: {mae1}, MAE2: {mae2},MAE3: {mae3}")
print(f"SPN-GP  NLPD: {nlpd2}")

