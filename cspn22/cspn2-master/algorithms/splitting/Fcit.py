'''
Created on January 14, 2019

@author: Alejandro Molina
'''
import sys
from itertools import combinations

from networkx import from_numpy_matrix, connected_components

from algorithms.splitting.RCoT import split_conditional_data_by_clusters
from structure.Conditional.utils import get_YX
import numpy as np
import fcit

epsilon = sys.float_info.epsilon


def fcit_conditional_test(i, j, y, x):
    return fcit.test(y[:,i], y[:,j], x)

def getFcitAdjacencyMatrix(y, x, n_jobs=1):
    n_features = y.shape[1]
    pairwise_comparisons = list(combinations(np.arange(n_features), 2))

    from joblib import Parallel, delayed

    rdc_vals = Parallel(n_jobs=n_jobs, max_nbytes=1024)(
        delayed(fcit_conditional_test)((i, j, y, x)) for i, j in pairwise_comparisons
    )

    rdc_adjacency_matrix = np.zeros((n_features, n_features))

    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc

    rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1

    return rdc_adjacency_matrix


def getCIGroup(alpha=0.9):
    # def getCIGroups(data, scope=None, alpha=0.001, families=None):
    def getCIGroups(local_data, ds_context=None, scope=None, families=None):
        """
        :param local_data: np array
        :param scope: a list of index to output variables
        :param alpha: threshold
        :param families: obsolete
        :return: np array of clustering

        This function take tuple (output, conditional) as input and returns independent groups
        alpha is the cutoff parameter for connected components
        """


        y, x = get_YX(local_data, ds_context.feature_size)

        pvals = getFcitAdjacencyMatrix(y, x) + epsilon

        pvals[pvals > alpha] = 0

        clusters = np.zeros(y.shape[1])
        for i, c in enumerate(connected_components(from_numpy_matrix(pvals))):
            clusters[list(c)] = i + 1

        return split_conditional_data_by_clusters(y, x, clusters, scope, rows=False)

    return getCIGroups