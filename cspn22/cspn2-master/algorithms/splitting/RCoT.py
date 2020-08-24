"""
Created on November 24, 2018

@author: Alejandro Molina
"""

import numpy as np
import os

from structure.Conditional.utils import get_YX, concatenate_yx
import rpy2.robjects.packages as rpackages

# utils = rpackages.importr('utils')
# utils.chooseCRANmirror(ind=1)
# utils.install_packages("devtools")
# DirichletReg = rpackages.importr("devtools")
path = os.path.dirname(__file__)
from networkx.algorithms.components.connected import connected_components
from networkx.convert_matrix import from_numpy_matrix
from rpy2 import robjects

from rpy2.robjects import numpy2ri

from rpy2.robjects.packages import SignatureTranslatedAnonymousPackage

import sys

epsilon = sys.float_info.epsilon

with open(path + "/RCoT.R", "r") as mixfile:
    code = "".join(mixfile.readlines())
    CoTest = SignatureTranslatedAnonymousPackage(code, "RCoT")


def split_conditional_data_by_clusters(y, x, clusters, scope, rows=True):
    assert not rows, "split conditional only for columns"

    nscope = np.asarray(scope)
    unique_clusters = np.unique(clusters)
    result = []

    for uc in unique_clusters:
        col_idx = clusters == uc
        local_data = concatenate_yx(y[:, col_idx].reshape((x.shape[0], -1)), x)
        proportion = 1
        result.append((local_data, nscope[col_idx].tolist(), proportion))
    return result


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
        BE CAREFUL WITH SPARSE DATA!
        """

        # data = preproc(local_data, ds_context, None, ohe)

        y, x = get_YX(local_data, ds_context.feature_size)

        pvals = testRcoT(y, x) + epsilon

        pvals[pvals > alpha] = 0

        clusters = np.zeros(y.shape[1])
        for i, c in enumerate(connected_components(from_numpy_matrix(pvals))):
            clusters[list(c)] = i + 1

        return split_conditional_data_by_clusters(y, x, clusters, scope, rows=False)

    return getCIGroups


def testRcoT(DataOut, DataIn):
    numpy2ri.activate()
    try:
        df_DataIn = robjects.r["as.data.frame"](DataIn)
        df_DataOut = robjects.r["as.data.frame"](DataOut)
        result = CoTest.testRCoT(df_DataOut, df_DataIn)
        result = np.asarray(result)
    except Exception as e:
        print(DataIn)
        print(np.sum(DataOut, axis=1))
        print(np.sum(DataOut, axis=0))
        0 / 0
        np.savetxt("/tmp/dataIn.txt", DataIn)
        np.savetxt("/tmp/dataOut.txt", DataOut)
        raise e

    return result
