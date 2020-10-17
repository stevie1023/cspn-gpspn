import numpy as np
from prod_structure import Mixture, Separator, GPMixture, Color
import gc
import torch
import gpytorch
from gpytorch.kernels import *
from gpytorch.likelihoods import *
from gpytorch.mlls import *
from torch.optim import *
from prod_structure import Product
import tensorflow as tf

y_d =3 ## indicates the scope of y

class Sum:
    def __init__(self, **kwargs):
        self.children = []
        self.weights = []
        self.scope=kwargs['scope']
        return None

    def forward(self, x_pred, **kwargs):
        if len(self.children) == 1:
            r_ = self.children[0].forward(x_pred, **kwargs)
            return r_[0].reshape(-1, 1), r_[1].reshape(-1, 1)
        _wei = np.array(self.weights).reshape((-1,1))

        a = self.children[0].forward(x_pred, **kwargs)[1] # mu(mean) of two children
        b= self.children[1].forward(x_pred, **kwargs)[1]
        c =self.children[0].forward(x_pred, **kwargs)[0] #cov(covariance) matrix for two children
        d = self.children[1].forward(x_pred, **kwargs)[0]

        mu_x = c*self.weights[0]+d*self.weights[1]
        co1=a*self.weights[0]+b*self.weights[1]

        t1 = tf.constant(c)
        t2 = tf.constant(c.transpose((0,2,1)))
        t3 = tf.matmul(t1,t2).eval(session=tf.compat.v1.Session())

        t4 = tf.constant(d)
        t5 = tf.constant(d.transpose((0, 2, 1)))
        t6 = tf.matmul(t4, t5).eval(session=tf.compat.v1.Session())

        co2 = t3*self.weights[0]+t6*self.weights[1]

        co3 =t3+t6

        co_x = co1+co2-co3 # covariance matrix of mixture distribution

        return mu_x, co_x

    def update(self):
        c_mllh = np.array([c.update() for c in self.children])
        new_weights = np.exp(c_mllh)
        self.weights = new_weights / np.sum(new_weights)

        return np.log(np.sum(new_weights))

    def __repr__(self, level=0, **kwargs):
        _wei = dict.get(kwargs, 'extra')
        _wei = Color.flt(_wei) if _wei else ""
        _sel = " " * (level) + f"{_wei} ✚ Sum"
        for i, child in enumerate(self.children):
            _wei = self.weights[i]
            _sel += f"\n{child.__repr__(level + 2, extra=_wei)}"

        return f"{_sel}"


class Split:
    def __init__(self, **kwargs):
        self.children = []
        self.split = kwargs['split']
        self.dimension = kwargs['dimension']
        self.depth = kwargs['depth']
        return None

    def forward(self, x_pred, **kwargs):
        smudge = dict.get(kwargs, 'smudge', 0)
        mu_x = np.zeros((len(x_pred),1, y_d))
        co_x = np.zeros((len(x_pred), y_d,y_d))
        left, right = self.children[0], self.children[1]

        smudge_scales = dict.get(kwargs, 'smudge_scales', {})
        smudge_scale = dict.get(smudge_scales, self.depth, 1)

        left_idx = np.where(x_pred[:, self.dimension] <= (self.split + smudge * smudge_scale))[0]
        right_idx = np.where(x_pred[:, self.dimension] > (self.split - smudge * smudge_scale))[0]
        # concatenate along axis=0
        mu_x[left_idx,:,:], co_x[left_idx,:,:] = left.forward(x_pred[left_idx], **kwargs)
        mu_x[right_idx,:,:], co_x[right_idx,:,:] = right.forward(x_pred[right_idx], **kwargs)

        return mu_x, co_x

    def update(self):
        return np.sum([c.update() for c in self.children])

    def __repr__(self, level=0, **kwargs):
        _wei = dict.get(kwargs, 'extra')
        _wei = Color.flt(_wei) if _wei else ""
        _spl = Color.val(self.split, f=1, color='yellow')
        _dim = Color.val(self.dimension, f=0, color='orange', extra="dim=")
        _dep = Color.val(self.depth, f=0, color='blue', extra="dep=")
        _sel = " " * (level - 1) + f"{_wei} ⓛ Split {_spl} {_dim} {_dep}"
        for child in self.children:
            _sel += f"\n{child.__repr__(level + 10)}"

        return f"{_sel}"

class Productt:
    def __init__(self, **kwargs):
        self.children = []
        self.scope = kwargs['scope']

        return None

    def forward(self, x_pred, **kwargs):
        mu_x = np.zeros((len(x_pred),1, y_d))
        co_x = np.zeros((len(x_pred), y_d,y_d ))

        if type(self.children[0]) is Sum:
            for i, child in enumerate(self.children):
                mu_c, co_c= child.forward(x_pred, **kwargs)
                mu_x += mu_c
                co_x += co_c
            return mu_x, co_x

        elif type(self.children[0]) is GP:
            for i, child in enumerate(self.children):
                mu_c, co_c = child.forward(x_pred, **kwargs)
                mu_x[:,0, child.scope], co_x[:, child.scope, child.scope] = mu_c.squeeze(-1), co_c.squeeze(-1)
            return mu_x, co_x


    def update(self):
        return np.sum([c.update() for c in self.children])

    def __repr__(self, level=0, **kwargs):
        _sel = " " * (level - 1) + f"ⓛ Product scope={self.scope} "
        for child in self.children:
            _sel += f"\n{child.__repr__(level + 10)}"

        return f"{_sel}"


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, **kwargs):
        x, y = kwargs['x'], kwargs['y']
        likelihood = kwargs['likelihood']
        gp_type = kwargs['type']
        xd = x.shape[1]

        active_dims = torch.tensor(list(range(xd)))

        super(ExactGPModel, self).__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if gp_type == 'mixed':
            m = MaternKernel(nu=0.5, ard_num_dims=xd, active_dims=active_dims)
            l = LinearKernel()
            self.covar_module = ScaleKernel(m + l)
            return
        elif gp_type == 'matern0.5':
            k = MaternKernel(nu=0.5)
        elif gp_type == 'matern1.5':
            k = MaternKernel(nu=1.5)
        elif gp_type == 'matern2.5':
            k = MaternKernel(nu=2.5)
        elif gp_type == 'rbf':
            k = RBFKernel()
        elif gp_type == 'matern0.5_ard':
            k = MaternKernel(nu=0.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'matern1.5_ard':
            k = MaternKernel(nu=1.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'matern2.5_ard':
            k = MaternKernel(nu=2.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'rbf_ard':
            k = RBFKernel(ard_num_dims=xd)
        elif gp_type == 'linear':
            k = LinearKernel(ard_num_dims=xd)  # ard_num_dims for linear doesn't actually work
        else:
            raise Exception("Unknown GP type")

        self.covar_module = ScaleKernel(k)


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    def __init__(self, **kwargs):
        self.type = kwargs['type']
        self.mins = kwargs['mins']
        self.maxs = kwargs['maxs']
        self.mll = 0
        self.x = dict.get(kwargs, 'x', [])
        self.scope = kwargs['scope']
        self.n = None
        self.y = dict.get(kwargs, 'y', [])
        self.count = kwargs['count']

    def forward(self, x_pred, **kwargs):

        mu_gp, co_gp= self.predict1(x_pred)
        return mu_gp, co_gp,

    def update(self):
        return self.mll  # np.log(self.n)*0.01 #-self.mll - 1/np.log(self.n)

    def predict1(self, X_s):
        self.model.eval()
        self.likelihood.eval()
        x = torch.from_numpy(X_s).float()  # .to('cpu') #.cuda()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            pm, pv = observed_pred.mean, observed_pred.variance
        x.detach()

        del x

        gc.collect()
        # modified
        return pm.detach().cpu(), pv.detach().cpu()

    def init(self, **kwargs):
        lr = dict.get(kwargs, 'lr', 0.20)
        steps = dict.get(kwargs, 'steps', 100)

        self.n = len(self.x)
        self.cuda = dict.get(kwargs, 'cuda') and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if self.cuda:
            torch.cuda.empty_cache()

        self.x = torch.from_numpy(self.x).float().to(self.device)
        self.y = torch.from_numpy(self.y.ravel()).float().to(self.device)


        # noises = torch.ones(self.n) * 1
        # self.likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        self.likelihood = GaussianLikelihood()
        # noise_constraint=gpytorch.constraints.LessThan(1e-2))
        # (noise_prior=gpytorch.priors.NormalPrior(3, 20))
        self.model = ExactGPModel(x=self.x, y=self.y, likelihood=self.likelihood, type=self.type).to(
            self.device)  # .cuda()
        self.optimizer = Adam([{'params': self.model.parameters()}], lr=lr)
        self.model.train()
        self.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        print(f"\tGP {self.type} init completed. Training on {self.device}")
        for i in range(steps):
            self.optimizer.zero_grad()  # Zero gradients from previous iteration
            output = self.model(self.x)  # Output from model
            loss = -mll(output, self.y)  # Calc loss and backprop gradients
            if i > 0 and i % 10 == 0:
                print(f"\t Step {i + 1}/{steps}, -mll(loss): {round(loss.item(), 3)}")

            loss.backward()
            self.optimizer.step()

        # LOG LIKELIHOOD NOW POSITIVE
        self.mll = -loss.detach().item()

        print(f"\tCompleted. +mll: {round(self.mll, 3)}")

        self.x.detach()
        self.y.detach()

        del self.x
        del self.y
        self.x = self.y = None

        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()

    def __repr__(self, level=0, **kwargs):
        _wei = dict.get(kwargs, 'extra')
        _wei = Color.flt(_wei) if _wei else ""

        _rou = 2
        _rng = [f"{round(self.mins[i], _rou)}-{round(self.maxs[i], _rou)}" for i, _ in enumerate(self.mins)]

        if self.n is not None:
            _cnt = Color.val(self.n, f=0, color='green', extra="n=")
        else:
            _cnt = 0

        _mll = Color.val(self.mll, f=3, color='orange', extra="mllh=")
        return " " * (level) + f"{_wei} ⚄ GP ({self.type}) {_rng} {_cnt} {_mll}"


def structure(root_region, scope,**kwargs):
    count=0
    root = Sum(scope=scope)
    to_process, gps = [(root_region, root)], dict()

    while len(to_process):
        gro, sto = to_process.pop()
        # sto = structure object

        if type(gro) is Mixture:
            for child in gro.children:
                if type(child) == Separator:
                    _child = Split(split=child.split, depth=child.depth, dimension=child.dimension)
                sto.children.append(_child)
            _cn = len(sto.children)
            sto.weights = np.ones(_cn) / _cn
            to_process.extend(zip(gro.children, sto.children))
        elif type(gro) is Separator: # sto is Split
            for child in gro.children:
                if type(child) is Product:
                    scope = child.scope
                    _child = Productt(scope = scope)
                    sto.children.append(_child)
                if type(child) is Mixture:
                    scope = child.scope
                    _child = Sum(scope=scope)
                    sto.children.append(_child)
            to_process.extend(zip(gro.children, sto.children))

        elif type(gro) is Product:
            i = 0
            for child in gro.children:
                if type(child) is Mixture:
                    scope = child.scope
                    _child = Sum(scope = scope)
                    sto.children.append(_child)
                else:
                    gp_type = 'rbf'
                    scopee = gro.scope
                    key = (*gro.mins, *gro.maxs, gp_type, count,scopee[i])
                    gps[key] = GP(type=gp_type, mins=gro.mins, maxs=gro.maxs,count = count, scope=scopee[i])
                    count+=1
                    sto.children.append(gps[key])
                    i += 1

            to_process.extend(zip(gro.children, sto.children))
    return root, list(gps.values())

