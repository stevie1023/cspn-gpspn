"""
Created on November 24, 2018

@author: Alejandro Molina
"""
import sys
sys.path.append('../')
from sklearn.cluster import KMeans
from spn.algorithms.splitting.Base import preproc, split_data_by_clusters
from spn.structure.StatisticalTypes import MetaType

from structure.Conditional.utils import get_YX

_rpy_initialized = False


import numpy as np

def init_rpy():
    global _rpy_initialized
    if _rpy_initialized:
        return
    _rpy_initialized = True

    from rpy2 import robjects
    from rpy2.robjects import numpy2ri
    import os

    path = os.path.dirname(__file__)
    with open(path + "/mixedClustering.R", "r") as rfile:
        code = "".join(rfile.readlines())
        robjects.r(code)

    numpy2ri.activate()

def get_split_conditional_rows_KMeans(n_clusters=2, pre_proc=None, ohe=False, seed=17):
    def split_conditional_rows_KMeans(local_data, ds_context, scope):
        y, x = get_YX(local_data, ds_context.feature_size)
        data = preproc(y, ds_context, pre_proc, ohe)

        clusters = KMeans(
            n_clusters=n_clusters, random_state=seed, precompute_distances=True
        ).fit_predict(data)

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_conditional_rows_KMeans


def get_split_rows_conditional_Gower(n_clusters=2, pre_proc=None, seed=17):
    from rpy2 import robjects

    init_rpy()

    def split_rows_Gower(local_data, ds_context, scope):
        y, x = get_YX(local_data, ds_context.feature_size)
        data = preproc(y, ds_context, pre_proc, False)


        feature_types = []
        for s in scope:
            mt = ds_context.meta_types[s]
            if mt == MetaType.BINARY:
                feature_types.append("categorical")
            elif mt == MetaType.DISCRETE:
                feature_types.append("discrete")
            else:
                feature_types.append("continuous")

        try:
            df = robjects.r["as.data.frame"](data)
            clusters = robjects.r["mixedclustering"](df, feature_types, n_clusters, seed)
            clusters = np.asarray(clusters)
        except Exception as e:
            np.savetxt("/tmp/errordata.txt", local_data)
            raise e

        return split_data_by_clusters(local_data, clusters, scope, rows=True)

    return split_rows_Gower