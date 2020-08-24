'''
Created on December 11, 2018

@author: Alejandro Molina
'''
from spn.structure.leaves.parametric.Parametric import Bernoulli, Gaussian, Poisson

from structure.Conditional.utils import get_X
import numpy as np


def leaf_predict_mode(node, data):
    return node.predictor.predict(get_X(data, node.feature_size))
