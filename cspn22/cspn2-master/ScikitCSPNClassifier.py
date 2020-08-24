"""
Created on November 25, 2018

@author: Alejandro Molina
"""
from sklearn.base import BaseEstimator, ClassifierMixin
from spn.algorithms.Inference import likelihood, log_likelihood
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli

from algorithms.ConditionalStructureLearning import learn_cspn_structure
from algorithms.ExactMPE import ExactMPE
from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from algorithms.splitting.Clustering import get_split_rows_conditional_Gower
from algorithms.splitting.RCoT import getCIGroup
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py
from structure.Conditional.Supervised import create_conditional_leaf
from algorithms.StructureLearning2 import create_leaf_node
from structure.Conditional.utils import concatenate_yx

import numpy as np


class CSPNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, parametric_types, exact_mpe=True, alpha=0, **kwargs):
        self.cspn = None
        self.context = None
        self.exact_mpe = exact_mpe
        self.kwargs = kwargs
        self.num_labels = 0
        self.alpha = alpha
        self.parametric_types = parametric_types

    def fit(self, X, y=None):
        self.context = Context(parametric_types=self.parametric_types).add_domains(y)
        self.context.feature_size = X.shape[1]
        self.num_labels = y.shape[1]

        def label_conditional(y, x):
            from sklearn.cluster import KMeans

            clusters = KMeans(
                n_clusters=2, random_state=17, precompute_distances=True
            ).fit_predict(x)
            return clusters

        self.cspn = learn_cspn_structure(
            concatenate_yx(y, X),
            self.context,
            split_rows=get_split_rows_conditional_Gower(),
            # split_rows=get_split_rows_KMeans(),
            # split_cols=get_split_cols_RDC_py(),
            split_cols=getCIGroup(alpha=self.alpha),
            # creeate_leaf = create_leaf_node,
            create_leaf=create_conditional_leaf,
            label_conditional=label_conditional,
            **self.kwargs
        )

        return self

    def predict(self, X, y=None):
        if self.cspn is None:
            raise RuntimeError("Classifier not fitted")

        y = np.array([np.nan] * X.shape[0] * len(self.cspn.scope)).reshape(
            X.shape[0], -1
        )

        test_data = concatenate_yx(y, X)

        mpe_y = ExactMPE(self.cspn, test_data, self.context)

        return mpe_y

    def predict_proba(self, X):
        y = np.ones((X.shape[0], self.num_labels))
        y[:] = np.nan

        test_data = concatenate_yx(y, X)

        results = np.ones_like(y)

        for n in self.cspn.scope:
            local_test = np.array(test_data)
            local_test[:, n] = 1
            results[:, n] = likelihood(self.cspn, local_test)[:, 0]

        return results

    def score_samples(self, X):
        return log_likelihood(self.cspn, X)
