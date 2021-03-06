import numpy as np
from collections import Counter
from scipy.stats import beta, iqr
import random
import sys
# from mzhu_code.RCoT import getCIGroup
import math
sys.path.append("/home/mzhu/madesi/mzhu_code/")
from RCoT import *

class Color():
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    LIGHTBLUE = '\033[96m'
    FADE = '\033[90m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def flt(flt):
        r = "%.4f" % flt
        return f"{Color.FADE}{r}{Color.ENDC}"

    @staticmethod
    def bold(txt):
        return f"{Color.OKGREEN}{txt}{Color.ENDC}"

    @staticmethod
    def val(flt, **kwargs):
        c = kwargs.get('color', 'yellow')
        e = kwargs.get('extra', '')
        f = kwargs.get('f', 4)

        if flt != float('-inf'):
            r = f"%.{f}f" % flt if flt != None else None
        else:
            r = '-∞'

        colors = {
            'yellow': Color.WARNING,
            'blue': Color.OKBLUE,
            'orange': Color.FAIL,
            'green': Color.OKGREEN,
            'lightblue': Color.LIGHTBLUE
        }
        return f"{colors.get(c)}{e}{r}{Color.ENDC}"


class Mixture:
    def __init__(self, **kwargs):
        self.maxs = kwargs['maxs']
        self.mins = kwargs['mins']
        self.deltas = dict.get(kwargs, 'deltas', [])
        self.spreads = self.maxs - self.mins
        self.dimension = dict.get(kwargs, 'dimension', None)
        self.children = dict.get(kwargs, 'children', [])
        self.depth = dict.get(kwargs, 'depth', 0)
        self.n = kwargs['n']
        self.scope = dict.get(kwargs, 'scope', [])
        self.parent = dict.get(kwargs, 'parent', None)
        self.splits = dict.get(kwargs, 'splits', [])  # for bins algo
        self.idx = dict.get(kwargs, 'idx', [])
        self.pop = dict.get(kwargs, 'pop', [])
        self.y = dict.get(kwargs, 'y', [])
        # assert np.all(self.spreads > 0)

    def __repr__(self, level=0):
        _dim = Color.val(self.dimension, f=0, color='orange', extra="dim=")
        _dep = Color.val(self.depth, f=0, color='yellow', extra="dep=")
        _nnn = Color.val(self.n, f=0, color='green', extra="n=")
        _rng = [f"{round(self.mins[i], 2)} - {round(self.maxs[i], 2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)

        if self.mins.shape[0] > 4:
            _rng = "..."

        _sel = " " * (level) + f"✚ Mixture [{_rng}] {_dim} {_dep} {_nnn}"

        if level <= 100:
            for split in self.children:
                _sel += f"\n{split.__repr__(level + 2)}"
        else:
            _sel += " ..."
        return f"{_sel}"


class Separator:
    def __init__(self, **kwargs):
        self.split = kwargs['split']
        self.splits = kwargs['splits']
        self.dimension = kwargs['dimension']
        self.depth = kwargs['depth']
        self.children = kwargs['children']
        self.parent = kwargs['parent']
        self.maxs = kwargs['maxs']
        self.mins = kwargs['mins']
        # self.y = kwargs['y']

    def __repr__(self, level=0):
        _sel = " " * (level) + f"ⓛ Separator dim={self.dimension} split={round(self.split, 2)}"

        for child in self.children:
            _sel += f"\n{child.__repr__(level + 2)}"

        return _sel


class Product:
    def __init__(self, **kwargs):
        self.children = kwargs['children']
        self.maxs = kwargs['maxsy']
        self.mins = kwargs['minsy']
        self.scope = dict.get(kwargs, 'scope', None)
        self.parent = dict.get(kwargs, 'parent', None)
        self.collect = dict.get(kwargs, 'collect', None)


    def __repr__(self, level=0):
        _nnn = Color.val(self.ny, f=0, color='green', extra="ny=")
        _rng = [f"{round(self.mins[i], 2)} - {round(self.maxs[i], 2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)

        if self.mins.shape[0] > 4:
            _rng = "..."

        _sel = " " * (level) + f"✚ Product y1={self.y1} y2 = {self.y2}[{_rng}]  {_nnn}"

        if level <= 100:
            for gp in self.children:
                _sel += f"\n{gp.__repr__(level + 2)}"
        else:
            _sel += " ..."
        return f"{_sel}"


class GPMixture:
    def __init__(self, **kwargs):
        self.mins = kwargs['mins']
        self.maxs = kwargs['maxs']
        self.idx = dict.get(kwargs, 'idx', [])
        self.parent = kwargs['parent']
        self.y = kwargs['y']

    def __repr__(self, level=0):
        _rng = [f"{round(self.mins[i], 2)} - {round(self.maxs[i], 2)}" for i, _ in enumerate(self.mins)]
        _rng = ", ".join(_rng)
        if self.mins.shape[0] > 4:
            _rng = "..."

        return " " * (level) + f"⚄ GPMixture [{_rng}] n={len(self.idx)}"


def _cached_gp(cache, **kwargs):
    min_, max_, y = list(kwargs['mins']), list(kwargs['maxs']), kwargs['y']
    cached = dict.get(cache, (*min_, *max_))
    if not cached:
        cache[(*min_, *max_)] = GPMixture(**kwargs)

    return cache[(*min_, *max_)]


def query(X, mins, maxs, skipleft=False):
    mask, D = np.full(len(X), True), X.shape[1]
    for d_ in range(D):
        if not skipleft:
            mask = mask & (X[:, d_] >= mins[d_]) & (X[:, d_] <= maxs[d_])
        else:
            mask = mask & (X[:, d_] >= mins[d_]) & (X[:, d_] <= maxs[d_])
    return np.nonzero(mask)[0]

def querym(X, mins, maxs,dm, skipleft=False):
    if skipleft:
        idx = np.where((X[:,dm]>mins[dm])&(X[:,dm]<=maxs[dm]))
    else:
        idx = np.where((X[:, dm] >= mins[dm]) & (X[:, dm] < maxs[dm]))

    return idx

def get_splits(X, dd, **kwargs):
    meta = dict.get(kwargs, 'meta', [""] * X.shape[1])
    max_depth = dict.get(kwargs, 'max_depth', 8)
    log = dict.get(kwargs, 'log', False)

    features_mask = np.zeros(X.shape[1])
    splits = np.zeros((X.shape[1], dd - 1))
    quantiles = np.quantile(X, np.arange(0, 1, 1 / dd)[1:], axis=0).T
    for i, var in enumerate(quantiles):
        include = False
        if dd == 2:
            spread = np.sum(X[:, i] < var[0]) - np.sum(X[:, i] >= var[0])

            if np.abs(spread) < X.shape[0] / 12:
                include = True
        elif len(np.unique(np.round(var, 8))) == len(var):
            include = True

        if include:
            features_mask[i] = 1
            splits[i] = np.array(var)

            if np.sum(features_mask) <= max_depth and meta and log:
                print(i, "\t", meta[i], var)
            else:
                pass  # print('.', end = '')

    return splits, features_mask


def build_bins(**kwargs):
    X = kwargs['X']

    Y = kwargs['Y']

    ddd = dict.get(kwargs, 'qd', 0)
    log = dict.get(kwargs, 'log', False)
    jump = dict.get(kwargs, 'jump', False)
    alphaa = dict.get(kwargs, 'alpha', False)
    min_idx = dict.get(kwargs, 'max_samples', 1)

    root_mixture_opts = {
        'mins': np.min(X, 0),
        'maxs': np.max(X, 0),
        'n': len(X),
        'scope': [i for i in range(Y.shape[1])],
        'parent': None,
        'dimension': np.argsort(-np.var(X, axis=0))[0],
        'idx': X,
        'y':Y
    }

    nsplits = Counter()
    root_node = Mixture(**root_mixture_opts)
    to_process, cache = [root_node], dict()
 # indicates the threshold when building leaves instead of sum nodes
    # the size of leaves is around min_dex/2
    count = 0
    while len(to_process):
        node = to_process.pop()
        if type(node) is Product:
            for i in range(len(node.children)):
                node2 = node.children[i]

                if type(node2) is Mixture :
                    d = node2.dimension
                    x_node = node2.idx

                    # y2 = node2.y
                    mins_node, maxs_node = np.min(x_node, 0), np.max(x_node, 0)
                    # r = beta.rvs(2, 2, size=x_node.shape[1])
                    # v = [(maxs_node[d] - mins_node[d]),
                    #      (maxs_node[d2] - mins_node[d2]),
                    #      (maxs_node[d3] - mins_node[d3])]

                    scope = node2.scope
                    d_selected = np.argsort(-np.var(x_node, axis=0))
                    d2 = d_selected[1]
                    # d3 = d_selected[2]
                    # d4 = d_selected[3]
                    #
                    # fit_lhs = node2.mins < splits[:, 0]
                    # fit_rhs = node2.maxs > splits[:, -1]
                    # create = np.logical_and(fit_lhs, fit_rhs)
                    # create = np.logical_and(create, features_mask)
                    # node_splits = []
                    # node_splits2 = []
                    # node_splits3 = []
                    # for node_split in splits[d]:
                    #     node_splits.append(node_split)
                    # for node_split in splits[d2]:
                    #     node_splits2.append(node_split)
                    # for node_split in splits[d3]:
                    #     node_splits3.append(node_split)
                    node_splits_all = [1, 2]
                    quantiles = np.quantile(x_node, np.linspace(0, 1, num = ddd+2), axis=0).T
                    d = [d, d2]
                    m = 0
                    for split in node_splits_all:
                        u = np.unique(quantiles[d[m]])
                        # u = quantiles[d[m]]
                        # if m == 0:
                        #     create_left = create.copy()
                        #     create_right = create.copy()
                        #     create_left[d[m]] = split != node_splits_all[0]
                        #     create_right[d[m]] = split != node_splits_all[
                        #         0]  # create left(right)nodes for splits other than the first/last split
                        # # no left nodes for the first split and no right nodes for the last split
                        #
                        # if m == 1:
                        #     create_left = create.copy()
                        #     create_right = create.copy()
                        #     create_left[d[m]] = split != node_splits_all[1]
                        #     create_right[d[m]] = split != node_splits_all[1]
                        #
                        # if m == 2:
                        #     create_left = create.copy()
                        #     create_right = create.copy()
                        #     create_left[d[m]] = split != node_splits_all[2]
                        #     create_right[d[m]] = split != node_splits_all[2]
                        # if jump:
                        #     # We force a new dimension for every child
                        #     # on the same split level
                        #     create_left[d[m]], create_right[d[m]] = False, False
                        #     create_right[np.argmax(create_left)] = False
                        # else:
                        #     # We dont create new mixture in the limits
                        #     create_left[d[m]] = split != node_splits[0]
                        #     create_right[d[m]] = split != node_splits[-1]
                        loop = []
                        # if len(u) == 1:
                        #     loop.append(x_node)
                        for i in range(len(u)-1):
                            new_maxs, new_mins = maxs_node.copy(), mins_node.copy()
                            skipleft = True
                            if i == 0:
                                skipleft = False

                            new_mins[d[m]] = u[i]
                            new_maxs[d[m]] = u[i + 1]

                            # new_mins[d[m]] = u[i] * 0.9 + 0.1 * (v[m] * r[d[m]] + mins_node[d[m]])
                            # new_maxs[d[m]] = u[i + 1] * 0.9 + 0.1 * (v[m] * r[d[m]] + mins_node[d[m]])
                            idx_i = query(x_node, new_mins, new_maxs, skipleft=skipleft)
                            if len(idx_i)==0:
                                print("idx=0")
                                continue
                            loop.append(idx_i)

                        # # for i in quantiles:
                        # new_maxs, new_mins = node2.maxs.copy(), node2.mins.copy()
                        # new_maxs[d[m]], new_mins[d[m]] = split, split
                        #
                        # new_maxs1, new_mins1 = node2.maxs.copy(), node2.mins.copy()
                        # new_maxs1[d[m]], new_mins1[d[m]] = q1[d[m]], q1[d[m]]
                        #
                        # new_maxs2, new_mins2 = node2.maxs.copy(), node2.mins.copy()
                        # new_maxs2[d[m]], new_mins2[d[m]] = q3[d[m]], q3[d[m]]
                        #
                        # idx_left = query(X, node2.mins, new_maxs1, skipleft=False)
                        # idx_middleft = query(X, new_mins1, new_maxs, skipleft=True)
                        # idx_middleright = query(X, new_mins, new_maxs2, skipleft=True)
                        # idx_right = query(X, new_mins2, node2.maxs, skipleft=True)
                        # print('left',len(idx_left))
                        # print('middleft',len(idx_middleft))
                        # print('middleright',len(idx_middleright))
                        # print('right',len(idx_right))
                        next_depth = node2.depth + 1
                        results = []
                        for idx in loop:
                            x_idx = x_node[idx]
                            maxs_loop = np.max(x_idx, axis=0)
                            mins_loop = np.min(x_idx, axis=0)
                            # y_idx = y2[idx]
                            next_dimension = np.argsort(-np.var(x_idx, axis=0))[0]
                            if len(scope) == 1:
                                if len(idx) < min_idx and len(idx)>0:
                                    gp = []
                                    prod_opts = {
                                        'minsy': mins_loop,
                                        'maxsy': maxs_loop,
                                        'scope': scope,
                                        'children': gp,
                                        # 'collect':collector
                                    }


                                    prod = Product(**prod_opts)
                                    a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, y=scope[0], parent=None)
                                    gp.append(a)
                                    results.append(prod)
                                else:
                                    mixture_opts = {
                                        'mins': mins_loop,
                                        'maxs': maxs_loop,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope,
                                        'idx': x_idx,
                                        # 'y':y_idx

                                    }
                                    results.append(Mixture(**mixture_opts))
                                    # divide = math.ceil(len(idx) / min_idx)
                                    # interval_ = np.min((math.ceil(len(idx) / divide), min_idx))
                                    # d_selected_ = np.argsort(-np.var(x_idx, axis=0))
                                    # d_ = 0
                                    #
                                    # result_sum = []
                                    # for j in range(2):
                                    #     gp_split = []
                                    #     results_split = []
                                    #     for i in range(divide):
                                    #         if i == divide - 1:
                                    #             idx_gp = x_idx[i * interval_:,:]
                                    #             idx_ = idx[i * interval_:]
                                    #             gp_split.append(idx_gp[d_selected_[d_]][0])
                                    #             gp_split.append(idx_gp[d_selected_[d_]][-1])
                                    #         else:
                                    #             idx_gp =x_idx[i * interval_:(i + 1) * interval_,:]
                                    #             idx_ = idx[i * interval_:(i + 1) * interval_]
                                    #             gp_split.append(idx_gp[d_selected_[d_]][0])
                                    #         idx_gp_max = np.max(idx_gp, axis=0)
                                    #         idx_gp_min = np.min(idx_gp, axis=0)
                                    #         a = _cached_gp(cache, mins=idx_gp_max, maxs=idx_gp_min, idx=idx_,
                                    #                        y=scope[0], parent=None)
                                    #
                                    #         gp = []
                                    #         gp.append(a)
                                    #         prod_opts = {
                                    #             'minsy': idx_gp_min,
                                    #             'maxsy': idx_gp_max,
                                    #             'scope': scope,
                                    #             'children': gp,
                                    #             'collect':collector
                                    #         }
                                    #         prod = Product(**prod_opts)
                                    #         results_split.append(prod)
                                    #         collector += 1
                                    #         idx_collection.append(idx_)
                                    #
                                    #     separator_opts = {
                                    #         'depth': node2.depth,
                                    #         'mins': mins_loop,
                                    #         'maxs': maxs_loop,
                                    #         'dimension': [d_selected_[d_]],
                                    #         'split': split,
                                    #         'children': results_split,
                                    #         'parent': None,
                                    #         'splits': gp_split
                                    #     }
                                    #     result_sum.append(Separator(**separator_opts))
                                    #     d_+=1
                                    #
                                    # mixture_opts = {
                                    #     'mins': mins_loop,
                                    #     'maxs': maxs_loop,
                                    #     'depth': next_depth,
                                    #     'dimension': next_dimension,
                                    #     'n': len(idx),
                                    #     'scope': scope,
                                    #     'idx': x_idx,
                                    #     'children':result_sum,
                                    #     'pop':1
                                    #     # 'y':y_idx
                                    # }
                                    # results.append(Mixture(**mixture_opts))

                            else:
                                # print('x',x_idx.shape)
                                # print('y',y_idx.shape)
                                # print('scope',scope2)
                                # cigroups = getCIGroup(x_idx,y_idx[:,scope2], scope=scope2, alpha=alphaa)
                                # print(cigroups)
                                # cigroup_all=[]
                                # scope_prod = []
                                # for sublist in cigroups:
                                #     for item in sublist:
                                #         scope_prod.append(item)
                                # if len(idx) >= min_idx:
                                #     for i,cigroup in enumerate(cigroups):
                                #         mixture_opts = {
                                #             'mins': mins_loop,
                                #             'maxs': maxs_loop,
                                #             'depth': next_depth,
                                #             'dimension': next_dimension,
                                #             'n': len(idx),
                                #             'scope': cigroup,
                                #             'idx': x_idx,
                                #             'y': y_idx
                                #         }
                                #         cigroup_all.append(Mixture(**mixture_opts))
                                #
                                #
                                #     prod_opts = {
                                #         'minsy': mins_loop,
                                #         'maxsy': maxs_loop,
                                #         'scope': scope_prod,
                                #         'children': cigroup_all
                                #     }
                                #     prod = Product(**prod_opts)
                                #     results.append(prod)
                                a = int(len(scope) / 2)
                                scope1 = random.sample(scope, a)
                                scope2 = list(set(scope) - set(scope1))
                                if len(idx) >= min_idx:
                                    mixture_opts1 = {
                                        'mins': mins_loop,
                                        'maxs': maxs_loop,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope1,
                                        'idx': x_idx
                                    }
                                    mixture_opts2 = {
                                        'mins': mins_loop,
                                        'maxs': maxs_loop,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope2,
                                        'idx': x_idx
                                    }
                                    prod_opts = {
                                        'minsy': mins_loop,
                                        'maxsy': maxs_loop,
                                        'scope': scope1 + scope2,
                                        'children': [Mixture(**mixture_opts1), Mixture(**mixture_opts2)]
                                    }

                                    prod = Product(**prod_opts)
                                    results.append(prod)
                                else:
                                    gp = []
                                    prod_opts = {
                                        'minsy': mins_loop,
                                        'maxsy': maxs_loop,
                                        'scope': scope1+scope2,
                                        # 'scope':scope_prod,
                                        'children': gp,
                                        # 'collect':collector
                                    }
                                    # collector += 1
                                    # idx_collection.append(idx)

                                    prod = Product(**prod_opts)
                                    for yi in prod.scope:
                                        a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, y=yi, parent=None)
                                        gp.append(a)
                                        count += 1
                                    results.append(prod)

                        if len(results) != 1:
                            print('n_splits',len(results))
                            to_process.extend(results)
                            separator_opts = {
                                'depth': node2.depth,
                                'mins': mins_node,
                                'maxs': maxs_node,
                                'dimension': d[m],
                                'split': split,
                                'children': results,
                                'parent': None,
                                'splits':quantiles[d[m]]
                            }
                            node2.children.append(Separator(**separator_opts))
                        elif len(results) == 1:
                            node2.children.extend(results)
                            to_process.extend(results)
                        else:
                            raise Exception('1')
                        m += 1

        elif type(node) is Mixture:

            d = node.dimension
            x_node = node.idx
            # u, c = np.unique(x_node[:,0], return_counts=True)
            # dup = u[c > 20000]
            # print('duplicated',dup)
            # y = node.y
            mins_node, maxs_node = np.min(x_node, 0), np.max(x_node, 0)

            scope = node.scope
            d_selected = np.argsort(-np.var(x_node, axis=0))
            d2 = d_selected[1]
            # d3 = d_selected[2]
            # d4 = d_selected[3]
            # r = beta.rvs(2, 2, size=x_node.shape[1])
            # v = [(maxs_node[d] - mins_node[d]),
            #      (maxs_node[d2] - mins_node[d2]),
            #      (maxs_node[d3] - mins_node[d3])]

            quantiles = np.quantile(x_node, np.linspace(0, 1, num = ddd+2),axis=0).T


            node_splits_all = [1,2]
            if len(node_splits_all) == 0: raise Exception('1')
            d = [d, d2]

            m = 0

            for split in node_splits_all:
                u = np.unique(quantiles[d[m]])
                # u = quantiles[d[m]]
                loop = []
                # if len(u) == 1:
                #     loop.append(x_node)

                for i in range(len(u)-1):
                    new_maxs, new_mins = maxs_node.copy(), mins_node.copy()
                    skipleft = True
                    if i == 0:
                        skipleft = False
                    # new_mins[d[m]] = u[i]*0.9+0.1*(v[m] * r[d[m]] + mins_node[d[m]])
                    # new_maxs[d[m]] = u[i+1]*0.9+0.1*(v[m] * r[d[m]] + mins_node[d[m]])
                    new_mins[d[m]] = u[i]
                    new_maxs[d[m]] = u[i + 1]
                    idx_i = query(x_node, new_mins, new_maxs, skipleft=skipleft)
                    if len(idx_i)==0:
                        print("idx=0")
                        continue
                    loop.append(idx_i)

                next_depth = node.depth + 1
                results = []
                for idx in loop:

                    x_idx = x_node[idx]
                    maxs_loop = np.max(x_idx,axis=0)
                    mins_loop = np.min(x_idx,axis=0)
                    # y_idx = y[idx]
                    next_dimension = np.argsort(-np.var(x_idx, axis=0))[0]
                    if len(scope) == 1:
                        if len(idx) < min_idx and len(idx) >0:
                            gp = []
                            prod_opts = {
                                'minsy': mins_loop,
                                'maxsy': maxs_loop,
                                'scope': scope,
                                'children': gp,
                                # 'collect':collector
                            }

                            prod = Product(**prod_opts)
                            a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, y=scope[0], parent=None)
                            gp.append(a)
                            results.append(prod)
                        else:


                            mixture_opts = {
                                'mins': mins_loop,
                                'maxs': maxs_loop,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope,
                                'idx': x_idx,
                                # 'y': y_idx

                            }
                            results.append(Mixture(**mixture_opts))

                    else:
                        # print('x', x_idx.shape)
                        # print('y', y_idx.shape)
                        # print('scope', scope)
                        #
                        # cigroups = getCIGroup(x_idx,y_idx[:,scope], scope=scope, alpha=alphaa)
                        # print(cigroups)
                        # cigroup_all=[]
                        # scope_prod = []
                        # for sublist in cigroups:
                        #     for item in sublist:
                        #         scope_prod.append(item)
                        # if len(idx) >= min_idx:
                        #     for i,cigroup in enumerate(cigroups):
                        #         mixture_opts = {
                        #             'mins': mins_loop,
                        #             'maxs': maxs_loop,
                        #             'depth': next_depth,
                        #             'dimension': next_dimension,
                        #             'n': len(idx),
                        #             'scope': cigroup,
                        #             'idx': x_idx,
                        #             'y': y_idx
                        #         }
                        #         cigroup_all.append(Mixture(**mixture_opts))
                        #
                        #     prod_opts = {
                        #         'minsy': mins_loop,
                        #         'maxsy': maxs_loop,
                        #         'scope':scope_prod ,
                        #         'children': cigroup_all
                        #     }
                        #
                        #     prod = Product(**prod_opts)
                        #     results.append(prod)

                        a = int(len(scope) / 2)
                        scope1 = random.sample(scope, a)
                        scope2 = list(set(scope) - set(scope1))
                        if len(idx) >= min_idx:
                            mixture_opts1 = {
                                'mins': mins_loop,
                                'maxs': maxs_loop,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope1,
                                'idx': x_idx
                            }
                            mixture_opts2 = {
                                'mins': mins_loop,
                                'maxs': maxs_loop,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope2,
                                'idx': x_idx
                            }
                            prod_opts = {
                                'minsy': mins_loop,
                                'maxsy': maxs_loop,
                                'scope': scope1+scope2,
                                'children': [Mixture(**mixture_opts1),Mixture(**mixture_opts2)]
                            }

                            prod = Product(**prod_opts)
                            results.append(prod)
                        else:
                            gp = []
                            prod_opts = {
                                'minsy': mins_loop,
                                'maxsy': maxs_loop,
                                'scope': scope1+scope2,
                                # 'scope':scope_prod,
                                'children': gp,
                                # 'collect':collector
                            }

                            prod = Product(**prod_opts)
                            for yi in prod.scope:
                                a = _cached_gp(cache, mins=mins_loop, maxs=maxs_loop, idx=idx, y=yi, parent=None)
                                gp.append(a)
                                count+=1
                            results.append(prod)



                if len(results) != 1:
                    print('n_splits', len(results))
                    to_process.extend(results)
                    separator_opts = {
                        'depth': node.depth,
                        'mins': mins_node,
                        'maxs': maxs_node,
                        'dimension': d[m],
                        'split': split,
                        'children': results,
                        'parent': None,
                        'splits':quantiles[d[m]]
                    }
                    node.children.append(Separator(**separator_opts))
                elif len(results) == 1:
                    node.children.extend(results)
                    to_process.extend(results)
                else:
                    raise Exception('1')
                m += 1

    gps = list(cache.values())
    aaa = [len(gp.idx) for gp in gps]
    c = (np.mean(aaa) ** 3) * len(aaa)
    r = 1 - (c / (len(X) ** 3))
    print("Full:\t\t", len(X) ** 3, "\nOptimized:\t", int(c), f"\n#GP's:\t\t {len(gps)} ({np.min(aaa)}-{np.max(aaa)})",
          "\nReduction:\t", f"{round(100 * r, 4)}%")
    print(f"nsplits:\t {nsplits}")
    print(f"Lengths:\t {aaa}\nSum:\t\t {sum(aaa)} (N={len(X)})")

    return root_node, gps
