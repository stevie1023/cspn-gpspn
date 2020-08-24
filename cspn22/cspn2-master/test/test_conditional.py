"""
Created on November 24, 2018

@author: Alejandro Molina
"""

import unittest

import numpy as np
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli
import sys
sys.path.append('../')
from ScikitCSPNClassifier import CSPNClassifier
from algorithms.ConditionalStructureLearning import learn_cspn_structure, create_conditional_slice
from algorithms.splitting.Clustering import get_split_conditional_rows_KMeans
from algorithms.splitting.RCoT import getCIGroup
from structure.Conditional.Inference import add_conditional_inference_support, conditional_supervised_likelihood
from structure.Conditional.Supervised import create_conditional_leaf
from structure.Conditional.utils import get_YX, concatenate_yx, get_Y


class TestConditional(unittest.TestCase):
    def setUp(self):
        add_conditional_inference_support()

    def test_utils(self):
        data = np.array([0, 1, 2, 3, 4, 5]).reshape(1, -1).repeat(10, axis=0)

        y, x = get_YX(data, 2)
        self.assertEqual(y.shape[0], 10)
        self.assertEqual(x.shape[0], 10)

        self.assertEqual(y.shape[1], 4)
        self.assertEqual(x.shape[1], 2)

        self.assertTrue(np.all(y[0, :] == [0, 1, 2, 3]))
        self.assertTrue(np.all(x[0, :] == [4, 5]))

        y, x = get_YX(data, 1)
        self.assertEqual(y.shape[0], 10)
        self.assertEqual(x.shape[0], 10)

        self.assertEqual(y.shape[1], 5)
        self.assertEqual(x.shape[1], 1)

        self.assertTrue(np.all(y[0, :] == [0, 1, 2, 3, 4]))
        self.assertTrue(np.all(x[0, :] == [5]))

        y, x = get_YX(data, 5)
        self.assertEqual(y.shape[0], 10)
        self.assertEqual(x.shape[0], 10)

        self.assertEqual(y.shape[1], 1)
        self.assertEqual(x.shape[1], 5)

        self.assertTrue(np.all(y[0, :] == [0]))
        self.assertTrue(np.all(x[0, :] == [1, 2, 3, 4, 5]))

    def test_conditional(self):
        labels = np.c_[np.zeros((500, 1)), np.ones((500, 1))]
        features = np.c_[
            np.r_[np.random.normal(5, 1, (500, 2)), np.random.normal(10, 1, (500, 2))]
        ]

        train_data = concatenate_yx(labels, features)

        ds_context = Context(
            parametric_types=[Bernoulli] * labels.shape[1]
        ).add_domains(labels)
        ds_context.feature_size = 2

        def label_conditional(y, x):
            from sklearn.cluster import KMeans

            clusters = KMeans(
                n_clusters=2, random_state=17, precompute_distances=True
            ).fit_predict(y)
            return clusters

        spn = learn_cspn_structure(
            train_data,
            ds_context,
            split_rows=get_split_conditional_rows_KMeans(),
            split_cols=getCIGroup(),
            create_leaf=create_conditional_leaf,
            label_conditional=label_conditional,
            cluster_univariate=True,
        )

    def test_conditional_node(self):
        data = np.r_[
            np.array([0] * 600).reshape(-1, 2), np.array([1] * 200).reshape(-1, 2)
        ]

        def label_conditional(y, x):
            return y

        node, data_slices = create_conditional_slice(data, 1, [0], label_conditional)

        self.assertAlmostEqual(data_slices[0][2], 0.75)
        self.assertAlmostEqual(data_slices[1][2], 0.25)

        y = get_Y(data, 1)

        children = [np.ones_like(y) * 2, np.ones_like(y) * 3]

        l = conditional_supervised_likelihood(node, children, data=data)

        self.assertTrue(np.all((y + 2) == l))

        children_data = np.array(range(1, 9)).reshape(-1, 2)
        X = np.array([0, 0, 1, 1, 0, 0, 1, 1]).reshape(-1, 2)
        l = conditional_supervised_likelihood(
            node,
            [children_data[:, 0].reshape(-1, 1), children_data[:, 1].reshape(-1, 1)],
            data=X,
        )
        self.assertTrue(np.all(l[:, 0] == [1, 4, 5, 8]))


    def test_classifier(self):
        labels = np.c_[np.zeros((500, 1)), np.ones((500, 1))]
        features = np.c_[
            np.r_[np.random.normal(5, 1, (500, 2)), np.random.normal(10, 1, (500, 2))]
        ]

        model = CSPNClassifier([Bernoulli]*2)
        model.fit(features, labels)
        model.predict(features)


if __name__ == "__main__":
    unittest.main()
