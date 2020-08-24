'''
Created on December 10, 2018

@author: Alejandro Molina
'''

from spn.algorithms.MPE import get_mpe_top_down_leaf, add_node_mpe
import numpy as np
from spn.structure.leaves.parametric.Parametric import Bernoulli

from structure.Conditional.Inference import supervised_leaf_likelihood, conditional_supervised_likelihood
from structure.Conditional.Mode import leaf_predict_mode
from structure.Conditional.Supervised import SupervisedLeaf, SupervisedOr
from structure.Conditional.utils import get_X





def supervised_leaf_bottom_up_mpe(node, data=None, dtype=np.float64):
    probs = supervised_leaf_likelihood(node, data=data, dtype=dtype)

    mpe_ids = np.isnan(data[:, node.scope[0]])

    mode_data = np.array(data[mpe_ids,])
    mode_data[:, node.scope] = leaf_predict_mode(node, data[mpe_ids, :])

    probs[mpe_ids] = supervised_leaf_likelihood(node, data=mode_data, dtype=dtype)

    return probs


def supervised_leaf_top_down_mpe(node, input_vals, data=None, lls_per_node=None):
    if len(input_vals) == 0:
        return None

    mpe_ids = np.isnan(data[input_vals, node.scope])

    mode_data = leaf_predict_mode(node, data[input_vals[mpe_ids],:])

    get_mpe_top_down_leaf(node, input_vals, data=data, mode=mode_data[:, 0])


def supervised_or_top_down_mpe(node, parent_result, data=None, lls_per_node=None, rand_gen=None):
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


def add_conditional_mpe_support():
    add_node_mpe(SupervisedLeaf, supervised_leaf_bottom_up_mpe, supervised_leaf_top_down_mpe)
    add_node_mpe(SupervisedOr, conditional_supervised_likelihood, supervised_or_top_down_mpe)
