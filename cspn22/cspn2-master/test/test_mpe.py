"""
Created on November 24, 2018

@author: Alejandro Molina
"""

import unittest

import numpy as np
from spn.algorithms.Inference import likelihood
from spn.algorithms.MPE import mpe
from spn.structure.Base import Context
# from new_base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli, Gaussian, Categorical,create_parametric_leaf

from ScikitCSPNClassifier import CSPNClassifier
from structure.Conditional.Inference import (
    add_conditional_inference_support,
)
from structure.Conditional.MPE import add_conditional_mpe_support
from structure.Conditional.Supervised import create_conditional_leaf
from structure.Conditional.utils import concatenate_yx


class TestMPE(unittest.TestCase):
    def setUp(self):
        add_conditional_inference_support()
        add_conditional_mpe_support()

    def test_leaf_mpe_gaussian(self):
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

        # leaf = create_conditional_leaf(data, ds_context, [0])
        leaf = create_parametric_leaf(data, ds_context, [0])

        res = mpe(leaf, np.array([np.nan, 10, 10]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 20.435226001909466)

        res = mpe(leaf, np.array([np.nan, 1, 1]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 59.4752193542575)

        res = mpe(leaf, np.array([np.nan, 1, 1, np.nan, 10, 10]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 59.4752193542575)
        self.assertAlmostEqual(res[1, 0], 20.435226001909466)

        with self.assertRaises(AssertionError):
            mpe(leaf, np.array([np.nan, 1, 1, np.nan, 10, 10, 5, 10, 10]).reshape(-1, 3))

    def test_leaf_mpe_bernoulli(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([10, 10], np.eye(2), 5000),
                np.random.multivariate_normal([1, 1], np.eye(2), 5000),
            ),
            axis=0,
        )
        y = np.array([0] * 5000 + [1] * 5000).reshape(-1, 1)

        # associates y=0 with X=[10,10]
        # associates y=1 with X=[1,1]

        data = concatenate_yx(y, x)

        ds_context = Context(parametric_types=[Bernoulli])
        ds_context.feature_size = 2

        leaf = create_conditional_leaf(data, ds_context, [0])

        res = mpe(leaf, np.array([np.nan, 10, 10]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 0)

        res = mpe(leaf, np.array([np.nan, 1, 1]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 1)

        res = mpe(leaf, np.array([np.nan, 1, 1, np.nan, 10, 10]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 1)
        self.assertAlmostEqual(res[1, 0], 0)

        with self.assertRaises(AssertionError):
            mpe(leaf, np.array([np.nan, 1, 1, np.nan, 10, 10, 5, 10, 10]).reshape(-1, 3))

    def test_leaf_mpe_conditional(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([10, 10], np.eye(2), 5000),
                np.random.multivariate_normal([1, 1], np.eye(2), 5000),
            ),
            axis=0,
        )
        y = np.array([0] * 5000 + [1] * 5000).reshape(-1, 1)

        # associates y=0 with X=[10,10]
        # associates y=1 with X=[1,1]

        data = concatenate_yx(y, x)

        cspn = CSPNClassifier([Bernoulli] * y.shape[1], min_instances_slice=4990, cluster_univariate=True)
        cspn.fit(x, y)

        res = mpe(cspn.cspn, np.array([np.nan, 10, 10]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 0)

        res = mpe(cspn.cspn, np.array([np.nan, 1, 1]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 1)

        res = mpe(cspn.cspn, np.array([np.nan, 1, 1, np.nan, 10, 10]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 1)
        self.assertAlmostEqual(res[1, 0], 0)

        with self.assertRaises(AssertionError):
            mpe(cspn.cspn, np.array([np.nan, 1, 1, np.nan, 10, 10, 5, 10, 10]).reshape(-1, 3))

    def test_leaf_mpe_conditional_categorical(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([20, 20], np.eye(2), 5000),
                np.random.multivariate_normal([10, 10], np.eye(2), 5000),
                np.random.multivariate_normal([1, 1], np.eye(2), 5000),
            ),
            axis=0,
        )
        y = np.array([2] * 5000 + [1] * 5000 + [0] * 5000).reshape(-1, 1)

        # associates y=0 with X=[10,10]
        # associates y=1 with X=[1,1]

        data = concatenate_yx(y, x)

        cspn = CSPNClassifier([Categorical] * y.shape[1], min_instances_slice=14990, cluster_univariate=True)
        cspn.fit(x, y)

        res = mpe(cspn.cspn, np.array([np.nan, 10, 10]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 1)

        res = mpe(cspn.cspn, np.array([np.nan, 1, 1]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 0)

        res = mpe(cspn.cspn, np.array([np.nan, 1, 1, np.nan, 10, 10]).reshape(-1, 3))
        self.assertAlmostEqual(res[0, 0], 0)
        self.assertAlmostEqual(res[1, 0], 1)

        with self.assertRaises(AssertionError):
            mpe(cspn.cspn, np.array([np.nan, 1, 1, np.nan, 10, 10, 5, 10, 10]).reshape(-1, 3))

    def test_leaf_mpe_conditional_multilabel(self):
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

        res = mpe(cspn.cspn, np.array([np.nan, np.nan, 10, 10]).reshape(-1, 4))
        self.assertAlmostEqual(res[0, 0], 0)

        res = mpe(cspn.cspn, np.array([np.nan, np.nan, 1, 1]).reshape(-1, 4))
        self.assertAlmostEqual(res[0, 0], 1)

        res = mpe(cspn.cspn, np.array([np.nan, np.nan, 1, 1, np.nan, np.nan, 10, 10]).reshape(-1, 4))
        self.assertAlmostEqual(res[0, 0], 1)
        self.assertAlmostEqual(res[1, 0], 0)

        with self.assertRaises(AssertionError):
            mpe(cspn.cspn, np.array([np.nan, np.nan, 1, 1, np.nan, np.nan, 10, 10, 5, 5, 10, 10]).reshape(-1, 4))


if __name__ == "__main__":
    unittest.main()
