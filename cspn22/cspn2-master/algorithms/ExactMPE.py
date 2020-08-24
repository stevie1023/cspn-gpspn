"""
Created on November 25, 2018

@author: Alejandro Molina
"""
from sklearn.utils.extmath import cartesian
# from spn.algorithms.Inference import log_likelihood,conditional_log_likelihood
from new_inference import log_likelihood,log_likelihood, sum_log_likelihood, prod_log_likelihood,likelihood
from tqdm import tqdm
from new_base import get_nodes_by_type,eval_spn_top_down
from spn.structure.Base import Product, Sum
from structure.Conditional.utils import get_YX, concatenate_yx, get_Y
from spn.algorithms.Marginalization import marginalize
import numpy as np

def merge_input_vals(l):
    return np.concatenate(l)


def mpe_prod(node, parent_result, data=None, lls_per_node=None, rand_gen=None):
    if parent_result is None:
        return None

    parent_result = merge_input_vals(parent_result)

    children_row_ids = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = parent_result

    return children_row_ids


def mpe_sum(node, parent_result, data=None, lls_per_node=None, rand_gen=None):
    if parent_result is None:
        return None

    parent_result = merge_input_vals(parent_result)

    w_children_log_probs = np.zeros((len(parent_result), len(node.weights)))
    for i, c in enumerate(node.children):
        # w_children_log_probs[:, i] = lls_per_node[parent_result, c.id] + np.log(node.weights[i])
        w_children_log_probs[:, i] = np.log(node.weights[i])

    max_child_branches = np.argmax(w_children_log_probs, axis=1)

    children_row_ids = {}

    for i, c in enumerate(node.children):
        children_row_ids[c] = parent_result[max_child_branches == i]

    return children_row_ids

def get_mpe_top_down_leaf(node, input_vals, data=None, mode=0):
    if input_vals is None:
        return None

    input_vals = merge_input_vals(input_vals)

    # we need to find the cells where we need to replace nans with mpes
    data_nans = np.isnan(data[input_vals, node.scope])

    n_mpe = np.sum(data_nans)

    if n_mpe == 0:
        return None

    data[input_vals[data_nans], node.scope] = mode

_node_top_down_mpe = {Product: mpe_prod, Sum: mpe_sum}
_node_bottom_up_mpe = {}
_node_bottom_up_mpe_log = {Sum: sum_log_likelihood, Product: prod_log_likelihood}


def ExactMPE(spn, data, ds_context,node_top_down_mpe=_node_top_down_mpe,
    node_bottom_up_mpe_log=_node_bottom_up_mpe_log):
    y, x = get_YX(data, ds_context.feature_size)

    result = np.array(y)
    nodes = get_nodes_by_type(spn)

    lls_per_node = np.zeros((data.shape[0], len(nodes)))
    # one pass bottom up evaluating the likelihoods
    # for i in range(data.shape[0]):


        # result[i, :] = exact_mpe_row(spn, y[i, :], x[i, :])
    # log_likelihood(spn, data, dtype=data.dtype,lls_matrix=lls_per_node)
    a = likelihood(spn, data, dtype=data.dtype, lls_matrix=lls_per_node)
    # a = concatenate_yx(result,x)


    # instance_ids = np.arange(data.shape[0])
    #
    # # one pass top down to decide on the max branch until it reaches a leaf, then it fills the nan slot with the mode
    # eval_spn_top_down(spn,eval_functions=node_top_down_mpe,  parent_result=instance_ids, data=data, lls_per_node=lls_per_node)

    # return data

    return a
