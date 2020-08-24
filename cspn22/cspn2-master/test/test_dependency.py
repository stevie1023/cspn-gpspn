import sys
import unittest

import numpy as np;
from spn.structure.Base import Context, Sum

from ScikitCSPNClassifier import CSPNClassifier
from algorithms.StructureLearning2 import create_product

np.random.seed(1)
from algorithms.splitting.RCoT import getCIGroup, testRcoT
from structure.Conditional.Inference import (
    add_conditional_inference_support,
    conditional_supervised_likelihood,
)
N = 200
class TestNodes(unittest.TestCase):

    def setUp(self, N=200):
        self.N = N
        add_conditional_inference_support()

    def test_dependency(self):
        x1 = np.random.randn(N,1)
        x2 = np.random.randn(N,1)
        z = x1 + x2 + np.random.randn(N, 1)

        pvals = testRcoT(np.concatenate((x1, x2), axis=1), z)
        print("pvals of conditional dependency is \n %s" % pvals)

    def test_independency(self):
        z = np.random.randn(N, 1)
        x1 = z + np.random.randn(N, 1)
        x2 = z + np.random.randn(N, 1)

        pvals = testRcoT(np.concatenate((x1, x2), axis=1), z)
        print("pvals of conditional independency is \n %s" % pvals)

    def test(self):
        # Three independent random variables
        z = np.random.randn(N, 1)
        x1 = np.random.randn(N, 1)
        x2 = np.random.randn(N, 1)

        pvals = testRcoT(np.concatenate((x1, x2), axis=1), z)
        print("pvals of independent variables is \n %s" % pvals)



ds_context = Context()
ds_context.feature_size = 1



class TestNodes2(unittest.TestCase):

    def setUp(self, N=200):
        self.N = N
        add_conditional_inference_support()

    def test_dependency(self):
        x1 = np.random.binomial(1,.5,size=(N,1))
        x2 = np.random.binomial(1,.5,size=(N,1))
        x3 = np.random.binomial(1,.5,size=(N,1))
        x4 = np.random.binomial(1,.5,size=(N,1))
        z = x1 + x2 + np.random.normal(loc=0, scale=0.01, size=(N, 1))
        local_data = np.concatenate((x1,x2,x3,x4,z), axis=1)
        parent = Sum()
        parent.children.append(None)

        pvals  = testRcoT(np.concatenate((x1, x2, x3, x4), axis=1), z) + sys.float_info.epsilon
        pvals_min = np.min(pvals[np.triu_indices(4,1)])
        pvals_max = np.max(pvals[np.triu_indices(4,1)])
        alpha = [pvals_min-sys.float_info.epsilon, (pvals_max-pvals_min)/2.0, pvals_max+sys.float_info.epsilon]
        result = [create_product(data=local_data, parent=parent, context=ds_context, scope=[0,1,2,3], split_cols=getCIGroup(alpha=alp), no_clusters=True) for alp in alpha]
        children_scope = [list(map(lambda r: r[1]['scope'], res)) for res in result]

    def test_independency(self):
        z = np.random.binomial(1, .5,size=(N,1))
        x1 = np.logical_not(z)
        x2 = z #np.logical_and(z, np.random.binomial(1,.5,size=(N,1)))
        x3 = np.random.binomial(1, .5, size=(N,1))

        local_data = np.concatenate((x1,x2,x3,z), axis=1)
        parent = Sum()
        parent.children.append(None)
        pvals  = testRcoT(np.concatenate((x1, x2, x3), axis=1), z) + sys.float_info.epsilon
        pvals_min = np.min(pvals[np.triu_indices(3,1)])
        pvals_max = np.max(pvals[np.triu_indices(3,1)])
        alpha = [pvals_min-sys.float_info.epsilon, (pvals_max-pvals_min)/2.0, pvals_max+sys.float_info.epsilon]
        result = [create_product(data=local_data, parent=parent, context=ds_context, scope=[0,1,2], split_cols=getCIGroup(alpha=alp), no_clusters=True) for alp in alpha]
        children_scope = [list(map(lambda r: r[1]['scope'], res)) for res in result]
        print(pvals)
        print(children_scope)


    # def test(self):
    #     # Three independent random variables
    #     z = np.random.randn(N, 1)
    #     x1 = np.random.binomial(size=(N,1))
    #     x2 = np.random.binomial(size=(N,1))
    #
    #     # local_data = np.concatenate((x1,x2,z), axis=1)
    #     # parent = Sum()
    #     # parent.children.append(None)
    #     # parent.children[0] = create_product(data=local_data, parent=parent, context=ds_context, scope=[0,1], split_cols=getCIGroup(alpha=0.001), no_clusters=True)
    #
    #     parent = CSPNClassifier(cluster_univariate=False, allow_sum_nodes=True, alpha=0.001).fit(z, np.concatenate((x1,x2), axis=1))
    #     TreeVisualization.plot_spn(parent, file_name="independent.png")



if __name__ == "__main__":
    unittest.main()



