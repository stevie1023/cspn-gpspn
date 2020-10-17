import numpy as np
from collections import Counter
from scipy.stats import beta, iqr
import random


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
        self.splits = dict.get(kwargs, 'splits', []),  # for bins algo
        self.idx = dict.get(kwargs, 'idx', [])
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
        self.dimension = kwargs['dimension']
        self.depth = kwargs['depth']
        self.children = kwargs['children']
        self.parent = kwargs['parent']
        self.maxs = kwargs['maxs']
        self.mins = kwargs['mins']

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
    min_samples = dict.get(kwargs, 'min_samples', 0)
    qd = dict.get(kwargs, 'qd', 0)
    log = dict.get(kwargs, 'log', False)
    jump = dict.get(kwargs, 'jump', False)

    root_mixture_opts = {
        'mins': np.min(X, 0),
        'maxs': np.max(X, 0),
        'n': len(X),
        'scope': [i for i in range(Y.shape[1])],
        'parent': None,
        'dimension': np.argsort(-np.var(X, axis=0))[0],
        'idx': X
    }

    nsplits = Counter()
    root_node = Mixture(**root_mixture_opts)
    to_process, cache = [root_node], dict()
    min_idx = 4000 # indicates the threshold when building leaves instead of sum nodes
    # the size of leaves is around min_dex/2
    count = 0
    while len(to_process):
        node = to_process.pop()
        if type(node) is Product:
            for i in range(len(node.children)):
                node2 = node.children[i]
                if type(node2) is Mixture:
                    d = node2.dimension
                    X = node2.idx
                    mins, maxs = np.min(X, 0), np.max(X, 0)
                    splits, features_mask = get_splits(X, qd, meta=dict.get(kwargs, 'meta', None), log=log)
                    scope = node2.scope
                    d_selected = np.argsort(-np.var(X, axis=0))
                    d2 = d_selected[1]
                    d3 = d_selected[2]

                    fit_lhs = node2.mins < splits[:, 0]
                    fit_rhs = node2.maxs > splits[:, -1]
                    create = np.logical_and(fit_lhs, fit_rhs)
                    create = np.logical_and(create, features_mask)
                    node_splits = []
                    node_splits2 = []
                    node_splits3 = []
                    for node_split in splits[d]:
                        node_splits.append(node_split)
                    for node_split in splits[d2]:
                        node_splits2.append(node_split)

                    for node_split in splits[d3]:
                        node_splits3.append(node_split)

                    q1 = np.percentile(X, 33, axis=0)
                    q3 = np.percentile(X, 66, axis=0)
                    node_splits_all = [np.median(node_splits), np.median(node_splits2)]

                    if len(node_splits_all) == 0: raise Exception('1')
                    d = [d, d2, d3]
                    i = 0
                    j = 0
                    m = 0
                    for split in node_splits_all:
                        if m == 0:
                            create_left = create.copy()
                            create_right = create.copy()
                            create_left[d[m]] = split != node_splits_all[0]
                            create_right[d[m]] = split != node_splits_all[
                                0]  # create left(right)nodes for splits other than the first/last split
                        # no left nodes for the first split and no right nodes for the last split

                        if m == 1:
                            create_left = create.copy()
                            create_right = create.copy()
                            create_left[d[m]] = split != node_splits_all[1]
                            create_right[d[m]] = split != node_splits_all[1]

                        if m == 2:
                            create_left = create.copy()
                            create_right = create.copy()
                            create_left[d[m]] = split != node_splits_all[2]
                            create_right[d[m]] = split != node_splits_all[2]
                        if jump:
                            # We force a new dimension for every child
                            # on the same split level
                            create_left[d[m]], create_right[d[m]] = False, False
                            create_right[np.argmax(create_left)] = False
                        else:
                            # We dont create new mixture in the limits
                            create_left[d[m]] = split != node_splits[0]
                            create_right[d[m]] = split != node_splits[-1]

                        new_maxs, new_mins = node2.maxs.copy(), node2.mins.copy()
                        new_maxs[d[m]], new_mins[d[m]] = split, split

                        idx_left = query(X, node2.mins, new_maxs, skipleft=False)
                        idx_right = query(X, new_mins, node2.maxs, skipleft=True)
                        print('left', len(idx_left))
                        print('right', len(idx_right))
                        next_depth = node2.depth + 1

                        loop = [
                            ('left', create_left, idx_left, node.mins, new_maxs),
                            ('right', create_right, idx_right, new_mins, node.maxs)
                        ]

                        results = []
                        for _, create_mixture, idx, mins, maxs, in loop:
                            if min_samples == 0:
                                min_samples = min(len(idx_left), len(idx_right)) + 1
                            x_idx = X[idx]
                            next_dimension = np.argsort(-np.var(x_idx, axis=0))[0]
                            if len(scope) == 1:
                                if len(idx) < min_idx:
                                    gp = []
                                    prod_opts = {
                                        'minsy': mins,
                                        'maxsy': maxs,
                                        'scope': scope,
                                        'children': gp,
                                    }

                                    prod = Product(**prod_opts)
                                    a = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, y=scope[0], parent=None)
                                    gp.append(a)
                                    results.append(prod)
                                else:
                                    mixture_opts = {
                                        'mins': mins,
                                        'maxs': maxs,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope,
                                        'idx': x_idx
                                    }
                                    results.append(Mixture(**mixture_opts))

                            else:
                                # random selection for scopes
                                scope1 = random.sample(scope, int(len(scope) / 2))
                                scope2 = list(set(scope) - set(scope1))
                                if len(idx) >= min_idx:
                                    mixture_opts1 = {
                                        'mins': mins,
                                        'maxs': maxs,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope1,
                                        'idx': x_idx
                                    }
                                    mixture_opts2 = {
                                        'mins': mins,
                                        'maxs': maxs,
                                        'depth': next_depth,
                                        'dimension': next_dimension,
                                        'n': len(idx),
                                        'scope': scope2,
                                        'idx': x_idx
                                    }
                                    prod_opts = {
                                        'minsy': mins,
                                        'maxsy': maxs,
                                        'scope': scope1 + scope2,
                                        'children': [Mixture(**mixture_opts1), Mixture(**mixture_opts2)],
                                    }

                                    prod = Product(**prod_opts)
                                    results.append(prod)
                                else:
                                    gp = []
                                    prod_opts = {
                                        'minsy': mins,
                                        'maxsy': maxs,
                                        'scope': scope1 + scope2,
                                        'children': gp,
                                    }

                                    prod = Product(**prod_opts)
                                    for yi in prod.scope:
                                        a = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, y=yi, parent=None)
                                        gp.append(a)
                                        count += 1
                                    results.append(prod)
                        j += 1
                        m += 1

                        if len(results) != 1:
                            to_process.extend(results)
                            separator_opts = {
                                'depth': node2.depth,
                                'mins': mins,
                                'maxs': maxs,
                                'dimension': d[i],
                                'split': split,
                                'children': results,
                                'parent': None
                            }
                            node2.children.append(Separator(**separator_opts))
                        elif len(results) == 1:
                            node2.children.extend(results)
                            to_process.extend(results)
                        else:
                            raise Exception('1')
                        i += 1




        elif type(node) is Mixture:
            d = node.dimension
            X = node.idx
            mins, maxs = np.min(X, 0), np.max(X, 0)
            splits, features_mask = get_splits(X, qd, meta=dict.get(kwargs, 'meta', None), log=log)
            scope = node.scope
            d_selected = np.argsort(-np.var(X, axis=0))
            d2 = d_selected[1]
            d3 = d_selected[2]

            fit_lhs = node.mins < splits[:, 0]
            fit_rhs = node.maxs > splits[:, -1]
            create = np.logical_and(fit_lhs, fit_rhs)

            create = np.logical_and(create, features_mask)
            print('create', create)

            # Preprocess splits
            node_splits = []
            node_splits2 = []
            node_splits3 = []
            for node_split in splits[d]:
                node_splits.append(node_split)
            for node_split in splits[d2]:
                node_splits2.append(node_split)

            for node_split in splits[d3]:
                node_splits3.append(node_split)

            q1 = np.percentile(X, 33, axis=0)
            q3 = np.percentile(X, 66, axis=0)
            node_splits_all = [np.median(node_splits), np.median(node_splits2)]

            if len(node_splits_all) == 0: raise Exception('1')
            d = [d, d2, d3]
            i = 0
            j = 0
            m = 0
            for split in node_splits_all:
                if m == 0:
                    create_left = create.copy()
                    create_right = create.copy()
                    create_left[d[m]] = split != node_splits_all[0]
                    create_right[d[m]] = split != node_splits_all[0]

                if m == 1:
                    create_left = create.copy()
                    create_right = create.copy()
                    create_left[d[m]] = split != node_splits_all[1]
                    create_right[d[m]] = split != node_splits_all[1]

                if m == 2:
                    create_left = create.copy()
                    create_right = create.copy()
                    create_left[d[m]] = split != node_splits_all[2]
                    create_right[d[m]] = split != node_splits_all[2]
                if jump:
                    # We force a new dimension for every child
                    # on the same split level
                    create_left[d[m]], create_right[d[m]] = False, False
                    create_right[np.argmax(create_left)] = False
                else:
                    # We dont create new mixture in the limits
                    create_left[d[m]] = split != node_splits[0]
                    create_right[d[m]] = split != node_splits[-1]

                new_maxs, new_mins = node.maxs.copy(), node.mins.copy()
                new_maxs[d[m]], new_mins[d[m]] = split, split

                idx_left = query(X, node.mins, new_maxs, skipleft=False)
                idx_right = query(X, new_mins, node.maxs, skipleft=True)
                print('left', len(idx_left))
                print('right', len(idx_right))
                next_depth = node.depth + 1

                loop = [
                    ('left', create_left, idx_left, node.mins, new_maxs),
                    ('right', create_right, idx_right, new_mins, node.maxs)
                ]

                results = []
                for _, create_mixture, idx, mins, maxs, in loop:
                    if min_samples == 0:
                        min_samples = min(len(idx_left), len(idx_right)) + 1

                    x_idx = X[idx]
                    next_dimension = np.argsort(-np.var(x_idx, axis=0))[0]
                    if len(scope) == 1:
                        if len(idx) < min_idx:
                            gp = []
                            prod_opts = {
                                'minsy': mins,
                                'maxsy': maxs,
                                'scope': scope,
                                'children': gp,
                            }

                            prod = Product(**prod_opts)
                            a = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, y=scope[0], parent=None)
                            gp.append(a)
                            results.append(prod)
                        else:
                            mixture_opts = {
                                'mins': mins,
                                'maxs': maxs,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope,
                                'idx': x_idx
                            }
                            results.append(Mixture(**mixture_opts))

                    else:
                        a = int(len(scope) / 2)
                        scope1 = random.sample(scope, a)
                        scope2 = list(set(scope) - set(scope1))
                        if len(idx) >= min_idx:
                            mixture_opts1 = {
                                'mins': mins,
                                'maxs': maxs,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope1,
                                'idx': x_idx
                            }
                            mixture_opts2 = {
                                'mins': mins,
                                'maxs': maxs,
                                'depth': next_depth,
                                'dimension': next_dimension,
                                'n': len(idx),
                                'scope': scope2,
                                'idx': x_idx
                            }
                            prod_opts = {
                                'minsy': mins,
                                'maxsy': maxs,
                                'scope': scope1 + scope2,
                                'children': [Mixture(**mixture_opts1), Mixture(**mixture_opts2)],
                            }

                            prod = Product(**prod_opts)
                            results.append(prod)
                        else:
                            gp = []
                            prod_opts = {
                                'minsy': mins,
                                'maxsy': maxs,
                                'scope': scope1 + scope2,
                                'children': gp,
                            }

                            prod = Product(**prod_opts)
                            for yi in prod.scope:
                                a = _cached_gp(cache, mins=mins, maxs=maxs, idx=idx, y=yi, parent=None)
                                gp.append(a)
                            results.append(prod)
                j += 1
                m += 1

                if len(results) != 1:
                    to_process.extend(results)
                    separator_opts = {
                        'depth': node.depth,
                        'mins': mins,
                        'maxs': maxs,
                        'dimension': d[i],
                        'split': split,
                        'children': results,
                        'parent': None
                    }
                    node.children.append(Separator(**separator_opts))
                elif len(results) == 1:
                    node.children.extend(results)
                    to_process.extend(results)
                else:
                    raise Exception('1')
                i += 1

    gps = list(cache.values())
    aaa = [len(gp.idx) for gp in gps]
    c = (np.mean(aaa) ** 3) * len(aaa)
    r = 1 - (c / (len(X) ** 3))
    print("Full:\t\t", len(X) ** 3, "\nOptimized:\t", int(c), f"\n#GP's:\t\t {len(gps)} ({np.min(aaa)}-{np.max(aaa)})",
          "\nReduction:\t", f"{round(100 * r, 4)}%")
    print(f"nsplits:\t {nsplits}")
    print(f"Lengths:\t {aaa}\nSum:\t\t {sum(aaa)} (N={len(X)})")

    return root_node, gps
