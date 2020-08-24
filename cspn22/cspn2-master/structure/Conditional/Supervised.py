"""
Created on November 24, 2018

@author: Alejandro Molina
"""
import sys
sys.path.append('../')
from new_base import Leaf
# from spn.structure.Base import Leaf
from spn.structure.StatisticalTypes import Type
from spn.structure.leaves.parametric.Parametric import (
    Gaussian,
    Categorical,
    create_parametric_leaf,
)
from structure.Base import Or
from structure.Conditional.ScikitLinearModel import CSPNLinearModel
from structure.Conditional.utils import get_YX

class SupervisedOr(Or):
    def __init__(self, children=None, feature_size=0):
        Or.__init__(self, children)
        self.splitting_algo = None
        self.feature_size = feature_size


class SupervisedLeaf(Leaf):
    def __init__(self, scope=None, predictor=None, parametric_type=None,feature_size=0):
        assert parametric_type is not None, 'parametric type is required, was None'
        Leaf.__init__(self,scope)
        self.parametric_type = parametric_type
        self.predictor = predictor
        self.feature_size = feature_size


def create_conditional_leaf(data, context, scope):
    assert len(scope) == 1, "scope of univariate parametric for more than one variable?"

    feature_size = context.feature_size
    parametric_type = context.parametric_types[scope[0]]

    #
    predictor = CSPNLinearModel(parametric_type=parametric_type, feature_size=feature_size)##modefy
    node = SupervisedLeaf( scope=scope, predictor=predictor, parametric_type=parametric_type,
                          feature_size=feature_size)
    # node = SupervisedLeaf(data=data,scope=scope, predictor=None, parametric_type=parametric_type, feature_size=feature_size)

    y, X = get_YX(data, node.feature_size)

    node.predictor.fit(X, y)

    return node
