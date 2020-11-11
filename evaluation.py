import numpy as np
import pandas as pd
import dill as pickle
from torch.utils.data import TensorDataset, DataLoader
import torch

with open('model.dill', 'rb') as file:
    root = pickle.load(file)
batch_size = 5000
x1 = pd.read_csv(file path)
y1= pd.read_csv(file path)
mu2,std2 = x1.mean(),x1.std()
mu4,std4 = y1.mean(),y1.std()
x1 = (x1-mu2)/std2 # test_x
y1 = (y1-mu4)/std4 #test_y
x1 = x1.iloc[:,:].values
y1 = y1.iloc[:,:].values
mu=[]
cov=[]

mu,cov = root.forward(x1, smudge=0)

mu_s1 = mu[:,0, 0]
mu_s2 = mu[:,0, 1]
sqe1 = (mu_s1 - y1[:,0]) ** 2
sqe2 = (mu_s2 - y1[:,1]) ** 2
rmse1 = np.sqrt(sqe1.sum() / len(y1))
rmse2 = np.sqrt(sqe2.sum() / len(y1))
mae1 = np.sqrt(sqe1).sum() / len(y1)
mae2 = np.sqrt(sqe2).sum() / len(y1)
nlpd1=0
for i in range(mu.shape[0]):
    sigma = np.sqrt(np.abs(np.linalg.det(cov[i,:,:])))
    # d1 = (test.iloc[i,:3].values.reshape((1,1,y_d))-mu[i,:,:]).reshape((1,y_d))
    d1 = (y1[i, :].reshape((1, 1, 2)) - mu[i, :, :]).reshape((1, 2))
    a = 1/(np.power((2*np.pi),y1.shape[1]/2)*sigma)
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

