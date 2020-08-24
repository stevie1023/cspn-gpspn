"""
Created on November 25, 2018

@author: Alejandro Molina
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from spn.algorithms.Inference import likelihood
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli

from algorithms.ConditionalStructureLearning import learn_cspn_structure
from algorithms.ExactMPE import ExactMPE
from algorithms.splitting.Clustering import get_split_conditional_rows_KMeans
from algorithms.splitting.RCoT import getCIGroup
from structure.Conditional.Supervised import create_conditional_leaf
from structure.Conditional.utils import concatenate_yx

import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri
robjects.numpy2ri.activate()

class RClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, name, packages, **kwargs):
        self.name = name
        self.kwargs = kwargs

        for package in packages:
            robjects.r['require'](package)

    def fit(self, X, y=None):
        data = robjects.r["as.data.frame"](robjects.r["cbind"](X, y))
        formula = "+".join(['V%s' % x for x in range(X.shape[1]+1, X.shape[1]+y.shape[1]+1)]) + "~ ."
        self.classifier = robjects.r[self.name](robjects.r['as.formula'](formula), data=data)
        return self

    def predict(self, X, y=None):
        data = robjects.r["as.data.frame"](X)
        predictions = robjects.r['predict'](self.classifier, data, type = "response")
        return np.round(np.array(predictions).T)

    def predict_proba(self, X):
        data = robjects.r["as.data.frame"](X)
        predictions = robjects.r['predict'](self.classifier, data, type="response")
        return np.array(predictions).T


