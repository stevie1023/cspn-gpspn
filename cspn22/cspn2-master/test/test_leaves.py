"""
Created on November 24, 2018

@author: Alejandro Molina
"""
import sys
sys.path.append('../')
import unittest

import numpy as np
from spn.algorithms.Inference import likelihood
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli, Gaussian, Categorical

from structure.Conditional.Inference import (
    add_conditional_inference_support,
)
from structure.Conditional.Supervised import create_conditional_leaf
from structure.Conditional.utils import concatenate_yx


def get_ll(leaf, vals):
    return likelihood(leaf, np.array(vals).reshape(1, -1))[0, 0]


class TestNodes(unittest.TestCase):
    def setUp(self):
        add_conditional_inference_support()

    def test_leaf_bernoulli(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([10, 10], np.eye(2), 500),
                np.random.multivariate_normal([1, 1], np.eye(2), 500),
            ),
            axis=0,
        )
        y = np.array([1] * 500 + [0] * 500).reshape(-1, 1)

        data = concatenate_yx(y, x)

        ds_context = Context(parametric_types=[Bernoulli])
        ds_context.feature_size = 2

        leaf = create_conditional_leaf(data, ds_context, [0])

        l = likelihood(leaf, data)
        neg_data = np.concatenate([1 - y, x], axis=1)
        lneg = likelihood(leaf, neg_data)

        np.testing.assert_array_almost_equal(l + lneg, 1.0)

        self.assertTrue(np.all(l >= 0.5))
        self.assertTrue(np.all(lneg < 0.5))

    def test_leaf_bernoulli_bootstrap(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([10, 10], np.eye(2), 100),
                np.random.multivariate_normal([1, 1], np.eye(2), 100),
            ),
            axis=0,
        )
        y = np.array([1] * 100 + [0] * 100).reshape(-1, 1)

        data = concatenate_yx(y, x)

        ds_context = Context(parametric_types=[Bernoulli])
        ds_context.feature_size = 2

        leaf = create_conditional_leaf(data, ds_context, [0])

        l = likelihood(leaf, data)
        neg_data = np.concatenate([1 - y, x], axis=1)
        lneg = likelihood(leaf, neg_data)

        np.testing.assert_array_almost_equal(l + lneg, 1.0)

        self.assertTrue(np.all(l >= 0.5))
        self.assertTrue(np.all(lneg < 0.5))

    def test_leaf_no_variance_bernoulli(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([10, 10], np.eye(2), 500),
                np.random.multivariate_normal([1, 1], np.eye(2), 500),
            ),
            axis=0,
        )
        y = np.array([1] * 1000).reshape(-1, 1)

        data = concatenate_yx(y, x)

        ds_context = Context(parametric_types=[Bernoulli])
        ds_context.feature_size = 2

        leaf = create_conditional_leaf(data, ds_context, [0])
        l = likelihood(leaf, data)
        self.assertTrue(np.all(l >= 0.5))

    def test_leaf_gaussian(self):
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

        self.assertFalse(np.any(np.isnan(likelihood(leaf, data))))

        self.assertGreater(get_ll(leaf, [20, 10, 10]), get_ll(leaf, [20, 1, 1]))
        self.assertGreater(get_ll(leaf, [60, 1, 1]), get_ll(leaf, [60, 10, 10]))
        self.assertAlmostEqual(get_ll(leaf, [60, 1, 1]), 0.3476232862652)
        self.assertAlmostEqual(get_ll(leaf, [20, 10, 10]), 0.3628922322773634)

    def test_leaf_no_variance_gaussian(self):
        np.random.seed(17)
        x = np.concatenate(
            (
                np.random.multivariate_normal([10, 10], np.eye(2), 500),
                np.random.multivariate_normal([1, 1], np.eye(2), 500),
            ),
            axis=0,
        )
        y = np.array([1] * 1000).reshape(-1, 1)

        data = concatenate_yx(y, x)

        ds_context = Context(parametric_types=[Gaussian])
        ds_context.feature_size = 2

        leaf = create_conditional_leaf(data, ds_context, [0])
        l = likelihood(leaf, data)
        self.assertEqual(np.var(l[:, 0]), 0)
        self.assertAlmostEqual(l[0, 0], 0.398942280401432)

        data[:, 0] = 2
        leaf = create_conditional_leaf(data, ds_context, [0])
        l = likelihood(leaf, data)
        self.assertEqual(np.var(l[:, 0]), 0)
        self.assertAlmostEqual(l[0, 0], 0.398942280401432)

        data3 = np.array(data)
        data3[:, 0] = 3
        leaf = create_conditional_leaf(data3, ds_context, [0])
        l = likelihood(leaf, data)
        self.assertAlmostEqual(np.var(l[:, 0]), 0)
        self.assertAlmostEqual(l[0, 0], 0.241970724519143)

    def test_leaf_categorical(self):
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

        l0 = likelihood(leaf, concatenate_yx(np.ones_like(y) * 0, x))
        l1 = likelihood(leaf, concatenate_yx(np.ones_like(y) * 1, x))
        l2 = likelihood(leaf, concatenate_yx(np.ones_like(y) * 2, x))

        np.testing.assert_array_almost_equal(l0 + l1 + l2, 1.0)

        self.assertTrue(np.all(l0[1000:1500] > 0.85))
        self.assertTrue(np.all(l0[0:1000] < 0.15))

        self.assertTrue(np.all(l1[500:1000] > 0.85))
        self.assertTrue(np.all(l1[0:500] < 0.15))
        self.assertTrue(np.all(l1[1000:1500] < 0.15))

        self.assertTrue(np.all(l2[0:500] > 0.85))
        self.assertTrue(np.all(l2[500:15000] < 0.15))


if __name__ == "__main__":
    unittest.main()
