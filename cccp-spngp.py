import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from learnspngp import query, build_bins
from spngp import structure
import sys



np.random.seed(58)

# data = pd.read_csv('/export/homebrick/home/mzhu/mzhu_code/Concrete_Data_Yeh.csv')
data = pd.read_csv('/home/mzhu/mzhu_code/data6dnormal.csv')
data_abnormal = pd.read_csv('/home/mzhu/mzhu_code/data6dabnormal.csv')
data = pd.DataFrame(data).dropna() # miss = data.isnull().sum()/len(data)
data_abnormal = pd.DataFrame(data_abnormal).dropna()
x_train = pd.read_csv('/home/mzhu/madesi/mzhu_code/protein/x_train.csv')
x_test = pd.read_csv('/home/mzhu/madesi/mzhu_code/protein/x_test.csv')
y_train = pd.read_csv('/home/mzhu/madesi/mzhu_code/protein/y_train.csv')
y_test = pd.read_csv('/home/mzhu/madesi/mzhu_code/protein/y_test.csv')
# x_train = pd.read_csv('/export/homebrick/home/mzhu/madesi/mzhu_code/x_learn.csv')
# x_test = pd.read_csv('/export/homebrick/home/mzhu/madesi/mzhu_code/x_test.csv')
# y_train = pd.read_csv('/export/homebrick/home/mzhu/madesi/mzhu_code/y_learn.csv')
# y_test = pd.read_csv('/export/homebrick/home/mzhu/madesi/mzhu_code/y_test.csv')
mu1,std1 = x_train.mean(),x_train.std()
mu2,std2 = x_test.mean(),x_test.std()
mu3,std3 = y_train.mean(),y_train.std()
mu4,std4 = y_test.mean(),y_test.std()
x_train = (x_train-mu1)/std1
x_test = (x_test-mu2)/std2
y_train = (y_train-mu3)/std3
y_test = (y_test-mu4)/std4
print(x_train.shape)
# dmean, dstd = data.mean(), data.std()
# data = (data-dmean)/dstd
# dmean1, dstd1 = data_abnormal.mean(), data_abnormal.std()
# data_abnormal = (data_abnormal-dmean1)/dstd1
#
# # GPSPN on full data
# train = data.sample(frac=0.8, random_state=58)
# test  = data.drop(train.index)
# train_abnormal = data_abnormal.sample(frac=0.8, random_state=58)
# test_abnormal  = data_abnormal.drop(train_abnormal.index)
# x, y = train.iloc[:, :-1].values, train.iloc[:, -1].values.reshape(-1,1)
x, y = x_train.iloc[:, :].values, y_train.iloc[:, :].values.reshape(-1,1)
opts = {
    'min_samples':          0,
    'X':                    x, 
    'qd':                   4, 
    'max_depth':            10,
    'max_samples':      6000,
    'log':               True,
    'jump':              True,
    'reduce_branching':  True
}
root_region, gps_ = build_bins(**opts)
#root_region, gps_ = build(X=x, delta_divisor=3, max_depth=2)
root, gps, gps1, gps2        = structure(root_region, gp_types=['rbf'])  #modified



for i, gp in enumerate(gps):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]

    print(f"Training GP set1 {i+1}/{len(gps)} ({len(idx)})" #modified
    gp.init(cuda=True)

##modified
for i, gp in enumerate(gps1):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]

    print(f"Training GP set2 {i+1}/{len(gps)} ({len(idx)})") #modified
    gp.init1(cuda=True)

for i, gp in enumerate(gps2):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]

    print(f"Training GP set3 {i+1}/{len(gps)} ({len(idx)})") #modified
    gp.init2(cuda=True)

root.update()


mll_=[]
mll_abnormal_=[]
cov = []
cov_abnormal =[]
RMSE_=[]
RMSE_abnormal_=[]
all_mll=[]
all_mll_abnormal=[]
all_rmse=[]
all_rmse_abnormal=[]
# for smudge in np.arange(0, 0.5, 0.05):
mu_s0, cov_s0, mll0 = root.forward(x_test.iloc[:2000, :].values,y_test.iloc[:2000,:].values, smudge=0)
mu_s2, cov_s2, mll2 = root.forward(x_test.iloc[2000:4000, :].values,y_test.iloc[2000:4000,:].values, smudge=0)
mu_s3, cov_s3, mll3 = root.forward(x_test.iloc[4000:6000, :].values,y_test.iloc[4000:6000,:].values, smudge=0)
mu_s4, cov_s4, mll4 = root.forward(x_test.iloc[6000:8000, :].values,y_test.iloc[6000:8000,:].values, smudge=0)
mu_s5, cov_s5, mll5 = root.forward(x_test.iloc[8000:10000, :].values,y_test.iloc[8000:10000,:].values, smudge=0)
mu_s6, cov_s6, mll6 = root.forward(x_test.iloc[10000:12000, :].values,y_test.iloc[10000:12000,:].values, smudge=0)
mu_s7, cov_s7, mll7 = root.forward(x_test.iloc[12000:, :].values,y_test.iloc[12000:,:].values, smudge=0)
# mu_s_abnormal, cov_s_abnormal, mll_abnormal = root.forward(test_abnormal.iloc[:, :-1].values, test_abnormal.iloc[:, -1].values, smudge=0)

    # mu_s = (mu_s.ravel() * dstd.iloc[-1]) + dmean.iloc[-1]
mu_s = np.concatenate((mu_s0,mu_s2,mu_s3,mu_s4,mu_s5,mu_s6))
mu_s = (mu_s.ravel() * std4.iloc[-1]) + mu4.iloc[-1]

    # mu_s_abnormal = (mu_s_abnormal.ravel() * dstd1.iloc[-1]) + dmean1.iloc[-1]

    # mu_t = (test.iloc[:, -1]*dstd.iloc[-1]) + dmean.iloc[-1]
    # mu_t_abnormal = (test_abnormal.iloc[:, -1] * dstd1.iloc[
mu_t = (y_test.iloc[-1] * std4.iloc[-1] ) + mu4.iloc[-1]
sqe = (mu_s - mu_t.values)**2

    # rmse = np.sqrt(sqe.sum() / len(test))
    # mae = np.sqrt(sqe).sum() / len(test)

rmse = np.sqrt(sqe.sum()/len(y_test))
mae = np.sqrt(sqe).sum() / len(y_test)
    # sqe_abnormal = (mu_s_abnormal - mu_t_abnormal.values) ** 2

    # rmse_abnormal = np.sqrt(sqe_abnormal.sum() / len(test))

mll_.append(np.mean(mll0))
mll_.append(np.mean(mll2))
mll_.append(np.mean(mll3))
mll_.append(np.mean(mll4))
mll_.append(np.mean(mll5))
mll_.append(np.mean(mll6))

    # mll_abnormal_.append(np.mean(mll_abnormal))



    # cov_abnormal.append(np.mean(cov_s_abnormal))
RMSE_.append(rmse)
    # RMSE_abnormal_.append(rmse_abnormal)

all_rmse =np.sqrt(sqe)
        # all_rmse_abnormal=np.sqrt(sqe_abnormal)
all_mll.extend(mll0)
all_mll.extend(mll2)
all_mll.extend(mll3)
all_mll.extend(mll4)
all_mll.extend(mll5)
all_mll.extend(mll6)

print('mll_normal:', mll_)
print('mll_abnormal:', mll_abnormal_)
print('RMSE_normal:', RMSE_)
print('RMSE_abnormal:', RMSE_abnormal_)
print('mae:',mae)

y1 =np.hstack(all_mll)
all_rmse_improved = np.concatenate((all_rmse, all_rmse_abnormal))

np.savetxt('all_rmse_protein.csv', [all_rmse_improved], delimiter=',')
#
np.savetxt('all_mll_protein.csv', y1, delimiter=',')
