'''
Created on December 11, 2018

@author: Alejandro Molina
'''
import numpy as np
from spn.algorithms.Sampling import add_node_sampling, add_leaf_sampling
from spn.structure.leaves.parametric.Parametric import Gaussian, Poisson, Bernoulli, Categorical
from spn.structure.leaves.parametric.utils import get_scipy_obj

from structure.Conditional.Supervised import SupervisedLeaf, SupervisedOr
from structure.Conditional.utils import get_X


def sample_supervised_leaf(node, n_samples, data, rand_gen):
    assert isinstance(node, SupervisedLeaf)
    assert n_samples > 0

    ps = node.predictor.predict_params(get_X(data, node.feature_size))

    if node.parametric_type == Categorical:
        X = np.zeros((n_samples))
        for i, p in enumerate(ps):
            X[i] = rand_gen.choice(node.predictor.classes, p=p, size=1)
        return X

    ps = ps[:,0]

    scipy_obj = get_scipy_obj(node.parametric_type)

    if node.parametric_type == Gaussian:
        params = {"loc": ps, "scale": np.ones_like(ps)*5}
    elif node.parametric_type == Poisson:
        params = {"mu": ps}
    elif node.parametric_type == Bernoulli:
        params = {"p": ps}
    else:
        raise Exception("Node parametric type unknown: " + str(node.parametric_type))

    X = scipy_obj.rvs(size=n_samples, random_state=rand_gen, **params)
    return X


def sample_supervised_or(node, parent_result, data=None, lls_per_node=None, rand_gen=None):
    if len(parent_result) == 0:
        return None

    branch = node.classifier.predict(get_X(data[parent_result], node.feature_size))

    children_row_ids = []

    for i, c in enumerate(node.children):
        idx = (branch == i)

        if np.sum(idx) > 0:
            children_row_ids.append(parent_result[idx])
        else:
            children_row_ids.append([])

    return children_row_ids

def add_conditional_sampling_support():
    add_leaf_sampling(SupervisedLeaf, sample_supervised_leaf)
    add_node_sampling(SupervisedOr, sample_supervised_or)
