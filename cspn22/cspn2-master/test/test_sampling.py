"""
Created on November 24, 2018

@author: Alejandro Molina
"""

import unittest

import numpy as np
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Gaussian, Bernoulli, Categorical

from ScikitCSPNClassifier import CSPNClassifier
from structure.Conditional.Inference import (
    add_conditional_inference_support,
)
from structure.Conditional.MPE import add_conditional_mpe_support
from structure.Conditional.Sampling import add_conditional_sampling_support
from structure.Conditional.Supervised import create_conditional_leaf
from structure.Conditional.utils import concatenate_yx


class TestSampling(unittest.TestCase):
    def setUp(self):
        add_conditional_inference_support()
        add_conditional_mpe_support()
        add_conditional_sampling_support()

    def test_leaf_sampling(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([10, 10], np.eye(2), 5000),
                np.random.multivariate_normal([1, 1], np.eye(2), 5000),
            ),
            axis=0,
        )
        y = np.array(np.random.normal(20, 2, 5000).tolist() + np.random.normal(60, 2, 5000).tolist()).reshape(-1, 1)

        # associates y=20 with X=[10,10]
        # associates y=60 with X=[1,1]

        data = concatenate_yx(y, x)

        ds_context = Context(parametric_types=[Gaussian])
        ds_context.feature_size = 2

        leaf = create_conditional_leaf(data, ds_context, [0])

        res = sample_instances(leaf, np.array([np.nan, 10, 10] * 1000).reshape(-1, 3), 17)
        self.assertAlmostEqual(np.mean(res[:, 0]), 20.456669723751173)

        res = sample_instances(leaf, np.array([np.nan, 1, 1] * 1000).reshape(-1, 3), 17)
        self.assertAlmostEqual(np.mean(res[:, 0]), 59.496663076099196)

        res = sample_instances(leaf, np.array([np.nan, 1, 1, np.nan, 10, 10] * 1000).reshape(-1, 3), 17)
        self.assertAlmostEqual(np.mean(res[::2, 0]), 59.546359637084564)
        self.assertAlmostEqual(np.mean(res[1::2, 0]), 20.452118792501008)

        with self.assertRaises(AssertionError):
            sample_instances(leaf, np.array([np.nan, 1, 1, np.nan, 10, 10, 5, 10, 10]).reshape(-1, 3), 17)

    def test_leaf_sampling_multilabel(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([10, 10], np.eye(2), 5000),
                np.random.multivariate_normal([1, 1], np.eye(2), 5000),
            ),
            axis=0,
        )
        y = np.concatenate(
            (np.array([0] * 5000 + [1] * 5000).reshape(-1, 1),
             np.array([1] * 5000 + [0] * 5000).reshape(-1, 1),
             ),
            axis=1,
        )

        # associates y0=0 with X=[10,10]
        # associates y0=1 with X=[1,1]
        # associates y1=1 with X=[10,10]
        # associates y1=0 with X=[1,1]

        data = concatenate_yx(y, x)

        cspn = CSPNClassifier([Bernoulli] * y.shape[1], min_instances_slice=4990, cluster_univariate=True)
        cspn.fit(x, y)

        res = sample_instances(cspn.cspn, np.array([np.nan, np.nan, 10, 10] * 1000).reshape(-1, 4), 17)
        self.assertAlmostEqual(np.unique(res[:, 0]), 0)
        self.assertAlmostEqual(np.unique(res[:, 1]), 1)

        res = sample_instances(cspn.cspn, np.array([np.nan, np.nan, 1, 1] * 1000).reshape(-1, 4), 17)
        self.assertAlmostEqual(np.unique(res[:, 0]), 1)
        self.assertAlmostEqual(np.unique(res[:, 1]), 0)

        res = sample_instances(cspn.cspn, np.array([np.nan, 0, 1, 1, np.nan, 1, 10, 10] * 1000).reshape(-1, 4), 17)
        self.assertAlmostEqual(np.unique(res[::2, 0]), 1)
        self.assertAlmostEqual(np.unique(res[1::2, 0]), 0)
        self.assertAlmostEqual(np.unique(res[::2, 1]), 0)
        self.assertAlmostEqual(np.unique(res[1::2, 1]), 1)

        with self.assertRaises(AssertionError):
            sample_instances(cspn.cspn, np.array([np.nan, 1, 1, 1, np.nan, 0, 10, 10, 1, 1, 10, 10]).reshape(-1, 4), 17)

    def test_leaf_sampling_categorical(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([20, 20], np.eye(2), 500),
                np.random.multivariate_normal([10, 10], np.eye(2), 500),
                np.random.multivariate_normal([1, 1], np.eye(2), 500),
            ),
            axis=0,
        )
        y = np.array([2] * 500 + [1] * 500 + [0] * 500).reshape(-1, 1)

        data = concatenate_yx(y, x)

        ds_context = Context(parametric_types=[Categorical])
        ds_context.feature_size = 2

        leaf = create_conditional_leaf(data, ds_context, [0])

        res = sample_instances(leaf, np.array([np.nan, 10, 10] * 1000).reshape(-1, 3), RandomState(17))
        self.assertAlmostEqual(np.mean(res[:, 0]), 1, 1)

        res = sample_instances(leaf, np.array([np.nan, 1, 1] * 1000).reshape(-1, 3), RandomState(17))
        self.assertAlmostEqual(np.mean(res[:, 0]), 0, 1)

        res = sample_instances(leaf, np.array([np.nan, 1, 1, np.nan, 10, 10] * 1000).reshape(-1, 3), RandomState(17))
        self.assertAlmostEqual(np.mean(res[::2, 0]), 0, 1)
        self.assertAlmostEqual(np.mean(res[1::2, 0]), 1, 1)

        with self.assertRaises(AssertionError):
            sample_instances(leaf, np.array([np.nan, 1, 1, np.nan, 10, 10, 5, 10, 10]).reshape(-1, 3), RandomState(17))
if __name__ == "__main__":
    unittest.main()
