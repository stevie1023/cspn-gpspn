# prepare data
import gc

import dill
import smp as smp
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpytorch.priors import GammaPrior
from prod_structure import query, build_bins
from prod_inference import structure, ExactGPModel

img_array = np.array(Image.open('baboon1.png'))
# load original image, shape (n, n, 3)
H,W,_ = img_array.shape
scale = 2
img_train = img_array[::scale, ::scale,:] # downsample the original image to shape(n/2, n/2, 3)
x_train=[]
y_train = []
for i in range(int(H/scale)):
    for j in range(int(W/scale)):
        x_train.append([i,j])
        y_train.append(img_train[i,j])

y_test =img_array.reshape((H,W,3))/255
y_test = y_test.reshape((H*W,1,3))

x = np.asarray(x_train)/H
# y = np.asarray(y_train-mu1)/std2# train_y
y = np.asarray(y_train)/255
y_d = y.shape[1]
x_test=[]
for i in range(H):
    for j in range(W):
        x_test.append([i*0.5,j*0.5])

x_test = np.asarray(x_test)/H
# train SPGPN
opts = {
        'min_samples': 0,
        'X': x,
        'Y': y,
        'qd': 1,
        'max_depth': 100,
        'max_samples': 510,
        'log': True,
        'jump': True,
        'reduce_branching': True
    }
root_region, gps_ = build_bins(**opts)
1
root, gps = structure(root_region,scope = [i for i in range(y.shape[1])], gp_types=['matern1.5_ard'])
lr = 0.1
steps = 150

# # #
for i, gp in enumerate(gps):
    idx = query(x, gp.mins, gp.maxs)
    gp.x = x[idx]
    y_scope = y[:,gp.scope]
    gp.y = y_scope[idx]
    print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
    gp.init(cuda=True,lr = lr,steps=steps)

# filename = 'graph1.dill'
# dill.dump(root, open(filename, 'wb'))
root.update()
# # interpolation
filename2 = 'graph2.dill'
dill.dump(root, open(filename2, 'wb'))

# with open("/home/mzhu/madesi/mzhu_code/graph1.dill", "rb") as dill_file:
#     root = dill.load(dill_file)
mu,_ = root.forward(np.asarray(x_test), smudge=0,y_d = y_d)
# print(mu.shape)

rmse = 0
mae = 0
for k in range(y.shape[1]):
    mu_s1 = mu[:,0, k]
    sqe1 = (mu_s1 - y_test[:,0,k]) ** 2
    rmse1 = np.sqrt(sqe1.sum() / len(y_test))
    mae1 = np.sqrt(sqe1).sum() / len(y_test)
    mae+=mae1
    rmse+=rmse1
print('rmse',rmse)
print('mae',mae)

for k in range(mu.shape[0]):

    mu[k, 0, 0] = mu[k, 0, 0]*255
    mu[k, 0, 1] = mu[k, 0, 1]*255
    mu[k, 0, 2] = mu[k, 0, 2]*255

mu = mu.astype(np.uint8).reshape((H,W,3))
im = Image.fromarray(mu)
im.save("rainbowwei1.png")


im = Image.fromarray(img_train)
im.save("pisaa1.png")

mu[::scale,::scale,:] = img_train
im = Image.fromarray(mu)
im.save("rainbowwe12.png")

