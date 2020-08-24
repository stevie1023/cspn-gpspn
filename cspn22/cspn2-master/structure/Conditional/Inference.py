"""
Created on November 24, 2018

@author: Alejandro Molina
"""
from scipy.special import logsumexp
from spn.algorithms.Inference import add_node_likelihood

from structure.Conditional.Supervised import SupervisedOr, SupervisedLeaf
import numpy as np

from structure.Conditional.utils import get_YX, concatenate_yx, get_X


# def conditional_supervised_likelihood(node, children, data=None, dtype=np.float64):
#     assert len(children) == 2
#     assert node.classifier is not None
#
#     llchildren = np.concatenate(children, axis=1)
#
#     branch = node.classifier.predict(get_X(data, node.feature_size))
#     result = llchildren[np.arange(len(llchildren)), branch].reshape(-1, 1)
#
#     assert result.shape[0] == data.shape[0]
#     assert result.shape[1] == 1
#
#     return result

def conditional_supervised_likelihood(node, children, data=None, dtype=np.float64):
    assert len(children) == 2
    assert node.classifier is not None

    llchildren = np.concatenate(children, axis=1)

    p = node.classifier.predict_proba(get_X(data, node.feature_size))

    result = logsumexp(llchildren, b=p, axis=1).reshape(-1, 1)

    assert result.shape[0] == data.shape[0]
    assert result.shape[1] == 1
    return result

def supervised_leaf_likelihood(node, data=None, dtype=np.float64):
    assert len(node.scope) == 1, node.scope

    y, x = get_YX(data, node.feature_size)
    y = y[:, node.scope]

    probs = np.ones((y.shape[0], 1), dtype=dtype)

    marg_ids = np.isnan(y[:, 0])

    if np.sum(~marg_ids) > 0:
        observations_data = concatenate_yx(y[~marg_ids], x[~marg_ids])

        probs[~marg_ids] = node.predictor.predict_proba(observations_data)

    probs[np.isclose(probs,0)] = 0.000000001

    return probs


def add_conditional_inference_support():
    add_node_likelihood(SupervisedOr, conditional_supervised_likelihood, conditional_supervised_likelihood)
    add_node_likelihood(SupervisedLeaf, supervised_leaf_likelihood)
