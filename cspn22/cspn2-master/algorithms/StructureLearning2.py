"""
Created on November 21, 2018

@author: Alejandro Molina
"""
import inspect

import tqdm
from pathos.multiprocessing import ProcessingPool as Pool
from collections import deque, ChainMap, defaultdict
from enum import Enum
from queue import Queue

from spn.algorithms.splitting.RDC import get_split_cols_RDC_py

from spn.algorithms.splitting.Clustering import get_split_rows_KMeans

from spn.structure.leaves.parametric.Parametric import (
    Gaussian,
    Categorical,
    create_parametric_leaf,
)

try:
    from time import perf_counter
except:
    from time import time

    perf_counter = time

import numpy as np

from spn.algorithms.TransformStructure import Prune, Compress
from spn.algorithms.Validity import is_valid
from spn.structure.Base import Product, Sum, assign_ids, Context
from new_base import assign_ids

class SplittingOperations(Enum):
    GET_NEXT_OP = 0
    CREATE_LEAF_NODE = 1
    CREATE_SUM_NODE = 2
    CREATE_PRODUCT_NODE = 3
    CREATE_CONDITIONAL_NODE = 4
    NAIVE_FACTORIZATION = 5
    REMOVE_UNINFORMATIVE_FEATURES = 6


def remove_non_informative_features(
        data=None, node_id=0, scope=None, **kwargs
):
    prod_node = Product()
    prod_node.scope = scope
    prod_node.id = node_id

    uninformative_features_idx = np.var(data[:, scope], 0) == 0
    zero_variance_rvs = [s for s in scope]
    result = []
    for idx, zero_var in enumerate(uninformative_features_idx):
        if not zero_var:
            continue
        prod_node.children.append(None)
        rv = scope[idx]
        data_slice = data[:, rv].reshape(-1, 1)
        result.append(
            (
                SplittingOperations.CREATE_LEAF_NODE,
                {
                    "data": data_slice,
                    "parent_id": node_id,
                    "pos": len(prod_node.children) - 1,
                    "scope": [rv],
                },
            )
        )
        del zero_variance_rvs[idx]
    assert len(result) > 0
    prod_node.children.append(None)
    result.append(
        (
            SplittingOperations.GET_NEXT_OP,
            {
                "data": data[:, zero_variance_rvs],
                "parent_id": node_id,
                "pos": len(prod_node.children) - 1,
                "scope": zero_variance_rvs,
            },
        )
    )
    return prod_node, result


def next_operation(
        data=None,
        parent=None,
        pos=0,
        scope=None,
        no_clusters=False,
        no_independencies=False,
        is_first=False,
        cluster_first=True,
        cluster_univariate=True,
        min_features_slice=1,
        min_instances_slice=6000,
        multivariate_leaf=False,
        **kwargs
):
    minimalFeatures = len(scope) == min_features_slice
    # minimalFeatures = data.shape[1] == min_features_slice
    minimalInstances = data.shape[0] <= min_instances_slice

    result_op = None
    result_params = {"data": data, "parent": parent, "pos": pos, "scope": scope}

    if minimalFeatures:
        if minimalInstances or no_clusters:
            result_op = SplittingOperations.CREATE_LEAF_NODE
        else:
            if cluster_univariate:
                result_op = SplittingOperations.CREATE_SUM_NODE
            else:
                result_op = SplittingOperations.CREATE_LEAF_NODE
    else:
        if np.any(np.var(data[:, scope], 0) == 0):
            result_op = SplittingOperations.REMOVE_UNINFORMATIVE_FEATURES

        elif minimalInstances or (no_clusters and no_independencies):
            if multivariate_leaf:
                result_op = SplittingOperations.CREATE_LEAF_NODE
            else:
                result_op = SplittingOperations.NAIVE_FACTORIZATION

        elif no_independencies:
            result_op = SplittingOperations.CREATE_SUM_NODE

        elif no_clusters:
            result_op = SplittingOperations.CREATE_PRODUCT_NODE

        elif is_first:
            if cluster_first:
                result_op = SplittingOperations.CREATE_SUM_NODE
            else:
                result_op = SplittingOperations.CREATE_PRODUCT_NODE
        else:
            result_op = SplittingOperations.CREATE_PRODUCT_NODE

    return (None, [(result_op, result_params)])


def create_leaf_node(
        data=None, node_id=0, context=None, scope=None, create_leaf=None, **kwargs
):
    # assert create_leaf is not None, "No create_leaf lambda"

    node =  create_leaf(data, context, scope)
    node.id = node_id
    #
    return node, None


def create_product(
        data=None,
        node_id=0,
        parent_id=0,
        pos=0,
        context=None,
        scope=None,
        split_cols=None,
        **kwargs
):
    assert split_cols is not None, "No split_cols lambda"
    assert scope is not None, "No scope"
    data_slices = split_cols(data, context, scope)

    result = []

    if len(data_slices) == 1:
        result.append(
            (
                SplittingOperations.GET_NEXT_OP,
                {
                    "data": data,
                    "parent_id": parent_id,
                    "pos": pos,
                    "no_independencies": True,
                    "scope": scope,
                },
            )
        )
        return None, result

    node = Product()
    node.scope.extend(scope)
    node.id = node_id

    for data_slice, scope_slice, _ in data_slices:
        assert isinstance(scope_slice, list), "slice must be a list"

        node.children.append(None)
        result.append(
            (
                SplittingOperations.GET_NEXT_OP,
                {
                    "data": data_slice,
                    "parent_id": node_id,
                    "pos": len(node.children) - 1,
                    "scope": scope_slice,
                },
            )
        )

    return node, result


def create_sum(
        data=None, node_id=0,
        parent_id=0,
        pos=0,
        context=None, scope=None, split_rows=None, **kwargs
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
    node.id = node_id
    node.scope.extend(scope)

    for data_slice, scope_slice, proportion in data_slices:
        assert isinstance(scope_slice, list), "slice must be a list"

        node.children.append(None)
        node.weights.append(proportion)
        result.append(
            (
                SplittingOperations.GET_NEXT_OP,
                {
                    "data": data_slice,
                    "parent_id": node_id,
                    "pos": len(node.children) - 1,
                    "scope": scope,
                },
            )
        )

    return node, result


def naive_factorization(data=None, node_id=0, scope=None, **kwargs):
    assert scope is not None, "No scope"

    prod_node = Product()
    prod_node.scope = scope
    prod_node.node_id = node_id

    result = []
    for rv in scope:
        prod_node.children.append(None)
        data_slice = data[:, rv].reshape(-1, 1)
        result.append(
            (
                SplittingOperations.CREATE_LEAF_NODE,
                {
                    "data": data_slice,
                    "parent_id": node_id,
                    "pos": len(prod_node.children) - 1,
                    "scope": [rv],
                },
            )
        )

    return prod_node, result


_op_lambdas = {
    SplittingOperations.GET_NEXT_OP: next_operation,
    SplittingOperations.CREATE_SUM_NODE: create_sum,
    SplittingOperations.CREATE_PRODUCT_NODE: create_product,
    SplittingOperations.CREATE_LEAF_NODE: create_leaf_node,
    SplittingOperations.NAIVE_FACTORIZATION: naive_factorization,
    SplittingOperations.REMOVE_UNINFORMATIVE_FEATURES: remove_non_informative_features,
}


class IdCounter(object):
    def __init__(self, initval=0):
        from multiprocessing import Value, Lock
        self.val = Value('i', initval)
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.val.value += 1
            return self.val.value

    def value(self):
        with self.lock:
            return self.val.value


def learn_structure(
        dataset, context, op_lambdas=_op_lambdas, prune=True, validate=True, compress=True, num_worker_threads=30,
        parallelized_ops=set([SplittingOperations.GET_NEXT_OP, SplittingOperations.CREATE_PRODUCT_NODE,
                              SplittingOperations.NAIVE_FACTORIZATION, SplittingOperations.CREATE_CONDITIONAL_NODE
                                 , SplittingOperations.REMOVE_UNINFORMATIVE_FEATURES]),
        **kwargs
):
    assert dataset is not None
    assert context is not None
    assert op_lambdas is not None

    # non_consecutive but monotonic counter

    id_counter = IdCounter()

    # root = Product()
    root = Sum()
    root.children.append(None)
    root.id = id_counter.increment()

    nodes = {root.id: root}

    tasks = Queue()
    tasks.put(
        (
            SplittingOperations.GET_NEXT_OP,
            {"data": dataset, "parent_id": root.id, "parent_type": type(root), "pos": 0,
             'node_id': id_counter.increment(), "is_first": True},
        )
    )

    def op_lambda_eval(params):
        next_op, context, all_params = params
        func = op_lambdas.get(next_op, None)
        assert func is not None, "No lambda function associated with operation: %s" % (next_op)
        if func == create_leaf_node:
            result = func(context=context, **all_params)
        else:
            result = func(context=context, **all_params)
        return (all_params['parent_id'], all_params['pos']), result

    def handle_op_lambdas_result(result):
        (parent_id, pos), (node, subtasks) = result
        if node is not None:
            nodes[parent_id].children[pos] = node
            nodes[node.id] = node

        if subtasks is not None:
            assert isinstance(subtasks, list)
            for e in subtasks:
                assert isinstance(e, tuple)
                tasks.put(e)

    parallelizable_tasks = []

    while True:

        while not tasks.empty():
            task = tasks.get()
            assert task is not None

            next_op, op_params = task

            all_params = ChainMap(op_params, kwargs)
            all_params['node_id'] = id_counter.increment()
            all_params['parent_type'] = type(nodes[all_params['parent_id']])

            if True or parallelized_ops is not None and next_op in parallelized_ops:

                op_result = op_lambda_eval((next_op, context, all_params))
                print("next_op", next_op, 'rows', all_params['data'].shape[0], 'scope', all_params['scope'])

                try:
                    for r in op_result[1][1]:
                        op, par = r
                        newp = dict(par)
                        x = newp['data']
                        del newp['data']
                        print('res op', op, 'data', x.shape, 'par', newp)
                except:
                    pass

                handle_op_lambdas_result(op_result)
            else:
                parallelizable_tasks.append((next_op, context, all_params))

        with Pool(num_worker_threads) as pool:
            results = pool.imap(op_lambda_eval, parallelizable_tasks)
            for r in tqdm.tqdm(results, total=len(parallelizable_tasks)):
                handle_op_lambdas_result(r)
            parallelizable_tasks.clear()

        if tasks.empty():
            break

    node = root.children[0]
    assign_ids(node)
    #
    # if compress:
    #     node = Compress(node)
    # if prune:
    #     node = Prune(node)
    # # if validate:
    #     valid, err = is_valid(node)
    #     assert valid, "invalid spn: " + err

    return node


if __name__ == "__main__":
    train_data = np.c_[
        np.r_[np.random.normal(5, 1, (500, 2)), np.random.normal(10, 1, (500, 2))],
        np.r_[np.zeros((500, 1)), np.ones((500, 1))],
    ]
    spn = learn_structure(
        train_data,
        Context(parametric_types=[Gaussian, Gaussian, Categorical]).add_domains(
            train_data
        ),
        scope=list(range(train_data.shape[1])),
        split_rows=get_split_rows_KMeans(),
        split_cols=get_split_cols_RDC_py(),
        create_leaf=create_parametric_leaf,
    )

    from spn.io.plot import TreeVisualization

    TreeVisualization.plot_spn(spn, file_name="tree_spn2.png")

