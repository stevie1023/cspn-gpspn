'''
Created on December 07, 2018

@author: Alejandro Molina
'''
from imblearn.over_sampling import RandomOverSampler
from numpy.random.mtrand import RandomState
from scipy.special._ufuncs import expit
from scipy.stats import bernoulli, norm, poisson
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.linear_model.base import LinearModel
from spn.structure.leaves.parametric.Parametric import Bernoulli, Gaussian, Poisson, Categorical
from structure.Conditional.Supervised import *
from structure.Conditional.utils import get_YX, concatenate_yx
from learnspnnn import *
from spngppp import *


def add_constant(X):
    result = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
    return result


def bernoulli_proba(y, params):
    return bernoulli.pmf(y, params)


def gaussian_proba(y, params):
    return norm.pdf(y, params)


def poisson_proba(y, params):
    return poisson.pmf(y, params)


def identity(x):
    return x



class CSPNLinearModel(LinearModel):
    def __init__(self, parametric_type, feature_size=0, rng=None,forward = None):
        assert parametric_type is not None
        assert parametric_type in [Bernoulli, Gaussian, Poisson, Categorical]
        self.coefficients = None
        self.predictor = None
        self.classes_ = None
        self.parametric_type = parametric_type
        self.feature_size = feature_size
        self.rng = rng
        self.forward = forward

        if self.parametric_type == Bernoulli:
            self.proba_func = bernoulli_proba
            self.link = expit
        elif self.parametric_type == Gaussian:
            self.proba_func = gaussian_proba
            self.link = identity
        elif self.parametric_type == Poisson:
            self.proba_func = poisson_proba
            self.link = np.exp

    def fit(self, X, y=None):
        # X = concatenate_yx(y0, X)
        # y, x = get_YX(data, feature_size)
        assert y.shape[1] == 1
        opts = {
            'min_samples': 0,
            'X': X,
            'qd': 4,
            'max_depth': 5,
            'max_samples': 1000,
            'log': True,
            'jump': True,
            'reduce_branching': True
        }
        root_region, gps_ = build_bins(**opts)
        root, gps = structure(root_region, gp_types=['rbf'])
        # root,  gps1, gps2 = structure(root_region, gp_types=['rbf'])
        for i, gp in enumerate(gps):
            idx = query(X, gp.mins, gp.maxs)
            gp.x, gp.y = X[idx], y[idx]

            print(f"Training GP set1 {i + 1}/{len(gps)} ({len(idx)})")  # modified
            gp.init()
        #
        # for i, gp in enumerate(gps1):
        #     idx = query(X, gp.mins, gp.maxs)
        #     gp.x, gp.y = X[idx], y[idx]
        #
        #     print(f"Training GP set1 {i + 1}/{len(gps1)} ({len(idx)})")  # modified
        #     gp.init1(cuda=True)
        #
        # for i, gp in enumerate(gps2):
        #
        #     idx = query(X, gp.mins, gp.maxs)
        #     gp.x, gp.y = X[idx], y[idx]
        #
        #     print(f"Training GP set2 {i + 1}/{len(gps2)} ({len(idx)})")  # modified
        #     gp.init2(cuda=True)

        root.update()

        assert y.shape[1] == 1

        all_equal = len(np.unique(y)) == 1
        if all_equal:
            self.coefficients = [0] * (X.shape[1] + 1)

            if self.parametric_type == Bernoulli:
                self.coefficients[0] = 100
                if y[0, 0] == 0:
                    self.coefficients[0] *= -1.0

            elif self.parametric_type == Gaussian:
                self.coefficients[0] = y[0]

            elif self.parametric_type == Poisson:
                if y[0, 0] == 0:
                    lambda_ = np.log(0.01)
                else:
                    lambda_ = np.log(y[0])
                self.coefficients[0] = lambda_

            elif self.parametric_type == Categorical:
                self.classes_ = np.array([y[0, 0]])

            return
        #
        # y = y[:, 0]
        #
        # if len(X) < 200:
        #     rng = self.rng
        #     if rng is None:
        #         rng = RandomState(17)
        #      #     ros = RandomOverSampler(random_state=rng)
        #     X, y = ros.fit_sample(X, y)
        #
        # X = add_constant(X)  # adds intercept

        if self.parametric_type == Bernoulli:
            predictor = LogisticRegression(C=1, max_iter=1000, fit_intercept=False, tol=1e-15, class_weight="balanced",
                                           solver='lbfgs')
            self.coefficients = predictor.fit(X, y).coef_[0, :].tolist()
        elif self.parametric_type == Gaussian:
            predictor = Ridge(alpha=0.1, fit_intercept=False, max_iter=1000)
            self.coefficients = root
            #predictor = sm.GLM(y, X, family=sm.families.Gaussian())
            #self.coefficients = predictor.fit_regularized(maxiter=1000, alpha=0.01).params.tolist()
        elif self.parametric_type == Poisson:
            predictor = sm.GLM(y, X, family=sm.families.Poisson())
            self.coefficients = predictor.fit(maxiter=1000).params.tolist()
        elif self.parametric_type == Categorical:
            self.predictor = LogisticRegression(C=1, max_iter=1000, multi_class="multinomial", fit_intercept=False,
                                                tol=1e-15, class_weight="balanced", solver='saga', n_jobs=1)
            self.predictor.fit(X, y)
            self.classes_ = self.predictor.classes_
            #self.coefficients = self.predictor.fit(X, y).coef_[0, :].tolist()
        else:
            raise Exception("Node parametric type unknown: " + str(self.parametric_type))

        return self

    @property
    def classes(self):
        return self.classes_

    def predict(self, data, y=None):
        # params = self.predict_params(data, y)

        if self.parametric_type == Gaussian:
            y,x = get_YX(data,feature_size=self.feature_size)
            # return params
            label = np.ones((len(y),1))
            return self.coefficients.forward(x, smudge=0)[0]

            # return self.coefficients.forward(x,label, smudge=0)[0]
        # elif self.parametric_type == Poisson:
        #     return np.round(params)
        # elif self.parametric_type == Bernoulli:
        #     pred_probs = np.concatenate((1 - params, params), 1)
        #     return np.argmax(pred_probs, 1).reshape(-1, 1)
        # elif self.parametric_type == Categorical:
        #     argmax_class = np.argmax(params, 1)
        #     return self.classes_[argmax_class].reshape(-1, 1)
        # else:
        #     raise Exception("Node parametric type unknown: " + str(self.parametric_type))

    def predict_params(self, X, y=None):
        dataIn = add_constant(X)


        if self.parametric_type == Categorical:

            if self.predictor is None:
                return np.ones((X.shape[0], 1))

            return self.predictor.predict_proba(dataIn)

        linpred = np.dot(dataIn, self.coefficients).reshape(-1, 1)

        return self.link(linpred)

    def predict_proba(self, data):
        y, X = get_YX(data, self.feature_size)

        params = self.predict_params(X, y)

        if self.parametric_type == Categorical:
            result = np.zeros((data.shape[0], 1))
            for j, c in enumerate(self.classes_):
                idx = y == c
                result[idx] = params[idx[:, 0], j]

            return result

        return self.proba_func(y, params)
