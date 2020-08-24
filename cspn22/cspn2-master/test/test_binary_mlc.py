import unittest

from spn.structure.leaves.parametric.Parametric import Bernoulli

from ScikitCSPNClassifier import CSPNClassifier
from data.cspn_dataset import get_binary_mask, get_binary_data
from structure.Conditional.Inference import add_conditional_inference_support
import numpy as np

from structure.Conditional.utils import concatenate_yx


class MyTestCase(unittest.TestCase):
    def setUp(self):
        add_conditional_inference_support()

    def eval_mlc(self, train, test, val):
        pass

    def test_datasets(self):
        # start with jester
        ds = "jester"
        ev = "ev80"
        name, features, validation, train, test, n_discrete, n_bernoulli = get_binary_data(ds)
        _, features_msk, validation_msk, train_msk, test_msk, n_discrete_msk, n_bernoulli_msk = get_binary_mask(ds, ev)
        col_msk = np.isnan(train_msk)[0]
        train_x, valid_x, test_x = train[:, ~col_msk], validation[:, ~col_msk], test[:, ~col_msk]
        train_y, valid_y, test_y = train[:, col_msk], validation[:, col_msk], test[:, col_msk]

        cspn = CSPNClassifier(parametric_types=[Bernoulli] * train_y.shape[1], alpha=0.0001, min_splitting_instances=3000,
                              min_clustering_instances=2000)
        cspn.fit(train_x, y=train_y)

        ll = cspn.score_samples(concatenate_yx(test_y, test_x))

        print(ll.mean())



if __name__ == '__main__':
    unittest.main()
