from spn.structure.Base import Product, Sum
import sys
sys.path.append('../')


# from structure.Conditional.Inference import supervised_leaf_likelihood, conditional_supervised_likelihood
from algorithms.ExactMPE import ExactMPE
# from spn.algorithms.Inference import likelihood
from new_inference import likelihood
from ScikitCSPNClassifier import CSPNClassifier
from structure.Conditional.Supervised import SupervisedOr
from structure.Conditional.utils import get_YX, concatenate_yx, get_X
import numpy as np
import pandas as pd
from spn.structure.Base import Context
from structure.Conditional.Supervised import create_conditional_leaf
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py

from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.structure.leaves.parametric.Parametric import (
    Gaussian,
    Categorical,
    create_parametric_leaf,
)
#

np.random.seed(58)
data = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/ccpp.csv')
# data = pd.read_csv('/home/mzhu/madesi/madesi/mzhu_code/20000normal.csv')
data = pd.DataFrame(data).dropna()
dmean, dstd = data.mean(), data.std()
data = (data-dmean)/dstd
print(data)
train = data.sample(frac=0.8, random_state=58)
test = data.drop(train.index)
y, x = train.iloc[:, :2].values, train.iloc[:, 2:].values
# y[:,[1,0]] =y[:,[0,1]]

y1,x1 = test.iloc[:, :2].values, test.iloc[:, 2:].values
# y1[:,[1,0]] =y1[:,[0,1]]

#
# x= pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/x_train.csv')
# x1 = pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/x_test.csv')
# y = pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/y_train.csv')
# y1= pd.read_csv('/home/mzhu/madesi/datasets/datasets/usflight/y_test.csv')
#
# mu1,std1 =x.mean(),x.std()
# mu2,std2 = x1.mean(),x1.std()
# mu3,std3 =y.mean(),y.std()
# mu4,std4 = y1.mean(),y1.std()
# x = (x-mu1)/std1
# x1 = (x1-mu2)/std2
# y = (y-mu3)/std3
# y1 = (y1-mu4)/std4
# x = x.iloc[:4000,:].values
# x1 = x1.iloc[:200,:].values
# y = y.iloc[:4000,:].values
# y1 = y1.iloc[:200,:].values

context = Context(parametric_types=[Gaussian]*y.shape[1]).add_domains(y)
min_instances_slice =  6000
alpha = 0.1

cspn = CSPNClassifier(parametric_types=[Gaussian] * y.shape[1],
                              cluster_univariate=True, min_instances_slice=min_instances_slice,
                              alpha=alpha,
                              allow_sum_nodes=True
                              )

cspn.fit(x, y)
spn = cspn.cspn


def predict_proba(self, X):
    y = np.ones((X.shape[0], self.num_labels))
    y[:] = np.nan

    test_data = concatenate_yx(y, X)

    results = np.ones_like(y)
    local_test = np.array(test_data)

    for n in range(2):
        local_test = np.array(test_data)
        local_test[:, n] = 1
        results[:, n] = likelihood(self.cspn, local_test)[:, 0]



    rbinc = np.zeros((X.shape[0], 2))
    rbinc[:, 0] = 1 - results[:, 0]
    rbinc[:, 1] = results[:, 0]
    return rbinc
    # return results


def predict(self, X, check_input=True):
    if self.cspn is None:
        raise RuntimeError("Classifier not fitted")

    y = np.array([np.nan] * X.shape[0] * len(self.cspn.scope)).reshape(X.shape[0], -1)

    test_data = concatenate_yx(y, X)

    mpe_y = ExactMPE(self.cspn, test_data, self.context)

    return mpe_y

# a = x1
# step = 20
# mu= []
# for i in [0,20,40,60,80,100,120,140,160,180]:
#     mu1 = predict(cspn,a[i:i+step])
#     mu.extend(mu1)
mu = predict(cspn,x1)
print(mu)
mu = np.array(mu)
# mu_s1 = mu[:,0].ravel()*std2[0]+mu2[0]
# mu_s2 = mu[:,1].ravel()*std2[1]+mu2[1]
# mu_t1 = y1[:,0]*std4[0]+mu4[0]
# mu_t2 = y1[:,1]*std4[1]+mu4[1]
mu_s1 = (mu[:,0].ravel() * dstd.iloc[0]) + dmean.iloc[0]
mu_s2 = (mu[:,1].ravel() * dstd.iloc[1]) + dmean.iloc[1]

mu_t1 = (y1[:, 0] * dstd.iloc[0]) + dmean.iloc[0]
mu_t2 = (y1[:, 1] * dstd.iloc[1]) + dmean.iloc[1]

sqe1 = (mu_s1 - mu_t1) ** 2
sqe2 = (mu_s2 - mu_t2) ** 2
rmse1 = np.sqrt(sqe1.sum() / len(y1))
rmse2 = np.sqrt(sqe2.sum() / len(y1))
mae1 = np.sqrt(sqe1).sum() / len(y1)
mae2 = np.sqrt(sqe2).sum() / len(y1)
np.savetxt('windmill1.csv', mu_s1, delimiter=',')
np.savetxt('windmill2.csv', mu_s2, delimiter=',')
np.savetxt('windmill1_.csv', mu_t1, delimiter=',')
np.savetxt('windmill2_.csv', mu_t2, delimiter=',')
print('rmse1:',rmse1)
print('rmse2:',rmse2)
print('mae1:',mae1)
print('mae2:',mae2)