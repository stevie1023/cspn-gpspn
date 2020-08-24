"""
Created on November 24, 2018

@author: Alejandro Molina
"""
from spn.structure.Base import Product, Sum
import sys
sys.path.append('../')
from algorithms.StructureLearning2 import (
    next_operation,
    SplittingOperations,
    create_product,
    create_leaf_node,
    learn_structure,
    create_sum,
)


from structure.Conditional.Supervised import SupervisedOr
from structure.Conditional.utils import get_YX, concatenate_yx, get_X
import numpy as np
import pandas as pd
from spn.structure.Base import Context
from structure.Conditional.Supervised import create_conditional_leaf
from spn.algorithms.splitting.RDC import get_split_cols_RDC_py

from spn.algorithms.splitting.Clustering import get_split_rows_KMeans
from spn.structure.leaves.parametric.Parametric import (
    Gaussian,
    Categorical,
    create_parametric_leaf,
)
def remove_non_informative_features(
        data=None,
        node_id=0,
        scope=None,
        context=0,
        uninformative_features_idx=None,
        **kwargs
):
    assert uninformative_features_idx is not None, "parameter uninformative_features_idx can't be None"

    prod_node = Product()
    prod_node.scope = scope
    prod_node.id = node_id

    y, x = get_YX(data, context.feature_size)

    non_zero_variance_rvs = []
    non_zero_variance_idx = []
    result = []
    for idx, zero_var in enumerate(uninformative_features_idx):
        rv = scope[idx]

        if not zero_var:
            non_zero_variance_rvs.append(rv)
            non_zero_variance_idx.append(idx)
            continue

        prod_node.children.append(None)
        data_slice = concatenate_yx(y[:, idx].reshape(-1, 1), x)
        result.append(
            (
                SplittingOperations.CREATE_LEAF_NODE,
                {
                    "data": data_slice,
                    "parent_id": prod_node.id,
                    "pos": len(prod_node.children) - 1,
                    "scope": [rv],
                },
            )
        )
    assert len(result) > 0
    if len(non_zero_variance_idx) > 0:
        prod_node.children.append(None)
        result.append(
            (
                SplittingOperations.GET_NEXT_OP,
                {
                    "data": concatenate_yx(data[:, non_zero_variance_idx], x),
                    "parent_id": prod_node.id,
                    "pos": len(prod_node.children) - 1,
                    "scope": non_zero_variance_rvs,
                },
            )
        )

    return prod_node, result


def next_operation(
        data=None,
        parent_id=0,
        parent_type=None,
        pos=0,
        scope=None,
        no_clusters=False,
        no_splitting=False,
        no_independencies=True,
        # is_first=False,
        is_first=True,
        cluster_first=True,
        cluster_univariate=False,
        # cluster_univariate=True,
        min_features_slice=1,
        min_splitting_instances=500,
        min_clustering_instances=500,
        context=None,
        allow_sum_nodes=True,
        allow_conditioning_nodes=True,
        remove_uninformative_features=False,
        **kwargs
):
    y, x = get_YX(data, context.feature_size)

    isMinimalFeatures = y.shape[1] <= min_features_slice
    isMinimalClusteringInstances = y.shape[0] <= min_clustering_instances
    isMinimalSplittingInstances = y.shape[0] <= min_splitting_instances

    result_params = {
        "data": data,
        "parent_id": parent_id,
        "pos": pos,
        "scope": scope,
        "no_clusters": no_clusters,
        "no_independencies": no_independencies,
    }

    uninformative_features = np.var(y, 0) == 0
    if remove_uninformative_features and np.any(uninformative_features):
        result_op = SplittingOperations.REMOVE_UNINFORMATIVE_FEATURES
        result_params["uninformative_features_idx"] = uninformative_features
        return None, [(result_op, result_params)]
    #
    # if not isMinimalSplittingInstances and not no_splitting:
    #     #split as much as you can
    #     result_op = SplittingOperations.CREATE_CONDITIONAL_NODE
    #     # result_op = SplittingOperations.CREATE_SUM_NODE
    #     return None, [(result_op, result_params)]


    if isMinimalFeatures:
        if isMinimalClusteringInstances or no_clusters:
            return None, [(SplittingOperations.CREATE_LEAF_NODE, result_params)]
        else:
            if cluster_univariate:
                return None, [(SplittingOperations.CREATE_SUM_NODE, result_params)]
            else:
                return None, [(SplittingOperations.CREATE_LEAF_NODE, result_params)]

    if isMinimalClusteringInstances or (no_clusters and no_independencies):
        return None, [(SplittingOperations.NAIVE_FACTORIZATION, result_params)]

    if no_independencies:
        return None, [(SplittingOperations.CREATE_SUM_NODE, result_params)]

    if no_clusters:
        return None, [(SplittingOperations.CREATE_PRODUCT_NODE, result_params)]

    if is_first:
        if cluster_first:
            return None, [(SplittingOperations.CREATE_SUM_NODE, result_params)]
        else:
            return None, [(SplittingOperations.CREATE_PRODUCT_NODE, result_params)]

    return None, [(SplittingOperations.CREATE_PRODUCT_NODE, result_params)]


def naive_factorization(
        data=None, node_id=0, context=None, scope=None, **kwargs
):
    assert scope is not None, "No scope"

    prod_node = Product()
    prod_node.scope = scope
    prod_node.id = node_id

    y, x = get_YX(data, context.feature_size)

    result = []
    for i, rv in enumerate(scope):
        prod_node.children.append(None)
        data_slice = concatenate_yx(y[:, i].reshape(-1, 1), x)
        result.append(
            (
                SplittingOperations.CREATE_LEAF_NODE,
                {
                    "data": data_slice,
                    "parent_id": prod_node.id,
                    "pos": len(prod_node.children) - 1,
                    "scope": [rv],
                },
            )
        )

    return prod_node, result


def create_conditional_slice(local_data, feature_size, scope, label_conditional):
    cluster_labels = label_conditional(*get_YX(local_data, feature_size))

    unique_labels = np.unique(cluster_labels)
    if len(unique_labels) == 1:
        return None, None

    assert len(unique_labels) == 2

    features = get_X(local_data, feature_size)

    assert features.shape[0] == cluster_labels.shape[0]

    node = SupervisedOr(feature_size=feature_size)
    node.scope.extend(scope)
    # from sklearn.linear_model import LogisticRegression
    #
    # node.classifier = LogisticRegression(
    #     C=1,
    #     max_iter=10000,
    #     fit_intercept=True,
    #     tol=1e-15,
    #     class_weight="balanced",
    #     solver="lbfgs",
    # )
    #
    # # if local_data.shape[0] < 1000:
    # #    idx = np.random.randint(low=0, high=local_data.shape[0], size=1000)
    # #    node.classifier.fit(features[idx], cluster_labels[idx])
    # # else:
    # node.classifier.fit(features, cluster_labels)
    #
    # slice_idx = node.classifier.predict(features)
    #
    # if len(np.unique(slice_idx)) == 1:
    #     return None, None
    #
    # data_slices = []
    # idx = slice_idx == 0
    # data_slices.append((local_data[idx, :], scope, np.sum(idx) / len(slice_idx)))
    #
    # idx = slice_idx == 1
    # data_slices.append((local_data[idx, :], scope, np.sum(idx) / len(slice_idx)))

    return node, data_slices


def create_conditional(
        data=None,
        node_id=0,
        parent_id=0,
        pos=0,
        context=None,
        scope=None,
        label_conditional=None,
        **kwargs
):
    assert label_conditional is not None, "No label_conditional lambda"
    assert scope is not None, "No scope"

    node, data_slices = create_conditional_slice(
        data, context.feature_size, scope, label_conditional
    )

    if data_slices is None:
        return None, [
            (
                SplittingOperations.GET_NEXT_OP,
                {
                    "data": data,
                    "parent_id": parent_id,
                    "pos": pos,
                    "no_clusters": True,
                    "scope": scope,
                },
            )
        ]

    node.id = node_id


    result = []
    for data_slice, scope_slice, proportion in data_slices:
        assert isinstance(scope_slice, list), "slice must be a list"

        node.children.append(None)
        result.append(
            (
                SplittingOperations.GET_NEXT_OP,
                {
                    "data": data_slice,
                    "parent_id": node.id,
                    "pos": len(node.children) - 1,
                    "scope": scope,
                },
            )
        )

    return node, result


def create_sum(
        data=None, node_id=0,
        parent_id=0,
        pos=0,
        context=None, scope=None, split_rows=None, split_on_sum=True, **kwargs
):
    assert split_rows is not None, "No split_rows lambda"
    assert scope is not None, "No scope"

    result = []

    data_slices = split_rows(data, context, scope)

    if len(data_slices) == 1:
        result.append(
            (
                SplittingOperations.GET_NEXT_OP,
                {
                    "data": data,
                    "parent_id": parent_id,
                    "pos": pos,
                    "no_clusters": True,
                    "scope": scope,
                },
            )
        )
        return result

    node = Sum()
    node.scope.extend(scope)
    node.id = node_id
    # assert parent.scope == node.scope

    for data_slice, scope_slice, proportion in data_slices:
        assert isinstance(scope_slice, list), "slice must be a list"

        child_data = data
        if split_on_sum:
            child_data = data_slice

        node.children.append(None)
        node.weights.append(proportion)
        result.append(
            (
                SplittingOperations.GET_NEXT_OP,
                {
                    "data": child_data,
                    "parent_id": node.id,
                    "pos": len(node.children) - 1,
                    "scope": scope,
                },
            )
        )

    return node, result


_conditional_op_lambdas = {
    SplittingOperations.GET_NEXT_OP: next_operation,
    SplittingOperations.CREATE_SUM_NODE: create_sum,
    SplittingOperations.CREATE_PRODUCT_NODE: create_product,
    SplittingOperations.CREATE_CONDITIONAL_NODE: create_conditional,
    SplittingOperations.CREATE_LEAF_NODE: create_leaf_node,
    # SplittingOperations.CREATE_LEAF_NODE: create_conditional_leaf,
    SplittingOperations.NAIVE_FACTORIZATION: naive_factorization,
    SplittingOperations.REMOVE_UNINFORMATIVE_FEATURES: remove_non_informative_features,
}


def learn_cspn_structure(
        train_data,
        ds_context,
        op_lambdas=_conditional_op_lambdas,
        split_rows=None,
        split_cols=None,
        create_leaf=None,
        **kwargs
):
    y, x = get_YX(train_data, ds_context.feature_size)
    return learn_structure(
        train_data,
        ds_context,
        compress=False,
        scope=list(range(y.shape[1])),
        op_lambdas=op_lambdas,
        split_rows=split_rows,
        split_cols=split_cols,
        create_leaf=create_leaf,
        **kwargs
    )

