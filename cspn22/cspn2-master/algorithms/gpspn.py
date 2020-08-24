import sys
sys.path.append('../')
from learnspngp import *
from spngp import *


class Node(object):
    def __init__(self):
        self.id = 0
        self.scope = []

    @property
    def name(self):
        return "%sNode_%s" % (self.__class__.__name__, self.id)

    @property
    def parameters(self):
        raise Exception("Not Implemented")

    def __repr__(self):
        return self.name



class Leaf(Node):
    def __init__(self, scope=None):
        Node.__init__(self)
        if scope is not None:
            if type(scope) == int:
                self.scope.append(scope)
            elif type(scope) == list:
                self.scope.extend(scope)
            else:
                raise Exception("invalid scope type %s " % (type(scope)))
class Parametric(Leaf):
    def __init__(self, type, scope=None):
        Leaf.__init__(self, scope=scope)
        self._type = type

    @property
    def type(self):
        return self._type

class Gaussian(Parametric):
    """
    Implements a univariate gaussian distribution with parameters
    \mu(mean)
    \sigma ^ 2 (variance)
    (alternatively \sigma is the standard deviation(stdev) and \sigma ^ {-2} the precision)
    """

    type = Type.REAL
    property_type = namedtuple("Gaussian", "mean stdev")

    def __init__(self, mean=None, stdev=None, scope=None):
        Parametric.__init__(self, type(self).type, scope=scope)

        # parameters
        self.mean = mean
        self.stdev = stdev

    @property
    def parameters(self):
        return __class__.property_type(mean=self.mean, stdev=self.stdev)

    @property
    def precision(self):
        return 1.0 / self.variance

    @property
    def variance(self):
        return self.stdev * self.stdev

class GPSPN(Node):
    def __init__(self, scope=None):
        Node.__init__(self)
        if scope is not None:
            if type(scope) == int:
                self.scope.append(scope)
            elif type(scope) == list:
                self.scope.extend(scope)
            else:
                raise Exception("invalid scope type %s " % (type(scope)))

    opts = {
        'min_samples': 0,
        'X': x,
        'qd': 4,
        'max_depth': 10,
        'max_samples': 6000,
        'log': True,
        'jump': True,
        'reduce_branching': True
    }
    root_region, gps_ = build_bins(**opts)
    # root_region, gps_ = build(X=x, delta_divisor=3, max_depth=2)
    root, gps, gps1, gps2 = structure(root_region, gp_types=['rbf'])
