# prepare data
import smp as smp
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gpytorch.priors import GammaPrior
from prod_learnspngp import query, build_bins
from prod_gp import structure, ExactGPModel
from gpytorch.kernels import *
from gpytorch.likelihoods import *
import random
from torch import optim
from torch.optim import *
from gpytorch.mlls import *
import torch
import scipy.misc as smp
img_array = np.array(Image.open('rainbow.jpg'))
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


std1, mu1= np.std(x_train,axis=0), np.mean(y_train,axis=0)
std2, mu2= np.std(y_train,axis=0), np.mean(x_train,axis=0)
x = np.asarray(x_train-mu2)/ std1  # normalized train_x
# x1 = x1_/std1 # test_x
y = np.asarray(y_train-mu1)/std2# train_y
# y1 = y1_-mu1 #test_y
y_d = y.shape[1]
x_test=[]
for i in range(H):
    for j in range(W):
        x_test.append([i*0.5,j*0.5])
x_test = np.asarray(x_test-mu2)/std1
# # train SPGPN
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

root, gps = structure(root_region,scope = [i for i in range(y.shape[1])], gp_types=['matern1.5_ard'])


lr = 0.1
steps = 10

for i, gp in enumerate(gps):
    idx = query(x, gp.mins, gp.maxs)
    gp.x = x[idx]
    y_scope = y[:,gp.scope]
    gp.y = y_scope[idx]
    print(f"Training GP {i + 1}/{len(gps)} ({len(idx)})")
    gp.init(cuda=True,lr = lr,steps=steps)
root.update()
# # interpolation

mu,_ = root.forward(np.asarray(x_test), smudge=0,y_d = y_d)
y_interpolated = np.zeros((H,W,3), dtype=np.uint8)
for k in range(mu.shape[0]):
    mu[k, 0, 0] = mu[k, 0, 0]*std2[0]+mu1[0]
    mu[k, 0, 1] = mu[k, 0, 1]*std2[1]+mu1[1]
    mu[k, 0, 2] = mu[k, 0, 2]*std2[2]+mu1[2]
    # mu_ = np.array((mu_s1,mu_s2,mu_s3))
    # y_interpolated[] = mu_.astype(np.uint8)
for i in range(H):
    for j in range(W):
        y_interpolated[i,j] =mu[i*int(W/10)+j]


# y1 = np.array(y_interpolated).reshape((int(H/10),int(W/10),3))
# pil_img = Image.fromarray(y_interpolated.astype(np.uint8)).convert('RGB')
pil_img = smp.toimage( y_interpolated )
pil_img.save('pisaa.png')

# MSE = mean_squared_error(y_test, img) # compare the interpolated image with the original one