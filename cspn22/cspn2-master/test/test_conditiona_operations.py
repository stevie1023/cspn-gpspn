import unittest

import numpy as np
from spn.structure.Base import Sum, Context, Product

from algorithms.ConditionalStructureLearning import naive_factorization, remove_non_informative_features, create_sum, \
    create_conditional
from algorithms.StructureLearning2 import SplittingOperations
from structure.Conditional.Inference import conditional_supervised_likelihood
from structure.Conditional.utils import get_YX, concatenate_yx


class TestConditionalOps(unittest.TestCase):

    def test_naive_factorization(self):
        np.random.seed(17)
        data = np.arange(0, 1000).reshape(-1, 8)

        parent = Sum()
        parent.children.append(None)

        ctx = Context()
        ctx.feature_size = 4

        scope = [1, 3, 4, 6]
        data2 = np.array(data)
        result = naive_factorization(data=data2, parent=parent, pos=0, context=ctx, scope=list(scope))

        self.assertListEqual(data.tolist(), data2.tolist())

        self.assertEqual(parent.children[0], result[0][1]['parent'])

        y, x = get_YX(data, 4)

        self.assertEqual(len(result), len(scope))
        for i, s in enumerate(scope):
            r = result[i]
            self.assertEqual(len(r), 2)
            self.assertEqual(r[0], SplittingOperations.CREATE_LEAF_NODE)
            self.assertEqual(type(r[1]['parent']), Product)
            self.assertEqual(r[1]['pos'], i)
            self.assertListEqual(r[1]['scope'], [s])
            self.assertListEqual(r[1]['data'].tolist(), concatenate_yx(y[:, i], x).tolist())

    def test_remove_non_informative_features(self):
        np.random.seed(17)
        data = np.arange(0, 1000).reshape(-1, 8)
        data[:, 1] = 1
        data[:, 3] = 3

        parent = Sum()
        parent.children.append(None)

        ctx = Context()
        ctx.feature_size = 4

        scope = [1, 3, 4, 6]
        data2 = np.array(data)

        y, x = get_YX(data, 4)

        uninformative_features_idx = np.var(y, 0) == 0
        result = remove_non_informative_features(data=data2, parent=parent, pos=0, context=ctx, scope=list(scope),
                                                 uninformative_features_idx=uninformative_features_idx)

        self.assertListEqual(data.tolist(), data2.tolist())

        self.assertEqual(len(parent.children[0].children), len(result))

        resulting_scopes = [[3], [6], [1, 4]]
        resulting_data_y = [y[:, 1], y[:, 3], y[:, [0, 2]]]

        for i, r in enumerate(result):
            self.assertEqual(len(r), 2)
            self.assertEqual(type(r[1]['parent']), Product)
            self.assertEqual(parent.children[0], r[1]['parent'])
            self.assertListEqual(r[1]['scope'], resulting_scopes[i])
            self.assertEqual(r[1]['pos'], i)

            self.assertListEqual(r[1]['data'].tolist(), concatenate_yx(resulting_data_y[i], x).tolist())

    def test_remove_all_non_informative_features(self):
        np.random.seed(17)
        data = np.ones((200, 8))
        for i in range(4):
            data[:, i] = i

        parent = Sum()
        parent.children.append(None)

        ctx = Context()
        ctx.feature_size = 4

        scope = [1, 3, 4, 6]
        data2 = np.array(data)

        y, x = get_YX(data, 4)

        uninformative_features_idx = np.var(y, 0) == 0
        result = remove_non_informative_features(data=data2, parent=parent, pos=0, context=ctx, scope=list(scope),
                                                 uninformative_features_idx=uninformative_features_idx)

        self.assertListEqual(data.tolist(), data2.tolist())

        self.assertEqual(len(parent.children[0].children), len(result))

        resulting_scopes = [[1], [3], [4], [6]]
        resulting_data_y = [y[:, 0], y[:, 1], y[:, 2], y[:, 3]]

        for i, r in enumerate(result):
            self.assertEqual(len(r), 2)
            self.assertEqual(type(r[1]['parent']), Product)
            self.assertEqual(parent.children[0], r[1]['parent'])
            self.assertListEqual(r[1]['scope'], resulting_scopes[i])
            self.assertEqual(r[1]['pos'], i)
            self.assertListEqual(r[1]['data'].tolist(), concatenate_yx(resulting_data_y[i], x).tolist())

    def test_create_sum_with_split(self):
        np.random.seed(17)
        data = np.arange(0, 1000).reshape(-1, 8)

        parent = Sum()
        parent.children.append(None)

        ctx = Context()
        ctx.feature_size = 4

        scope = [1, 3, 4, 6]
        data2 = np.array(data)

        K = int(data.shape[0] * 0.25)
        split_idx = np.array([0] * K + [1] * (data.shape[0] - K))
        np.random.shuffle(split_idx)

        def split_rows(data, context, scope):
            result = []
            result.append((data[split_idx == 0, :], scope, 0.25))
            result.append((data[split_idx == 1, :], scope, 0.75))
            return result

        result = create_sum(data=data2, parent=parent, pos=0, context=ctx, scope=list(scope),
                            split_rows=split_rows, split_on_sum=True)

        self.assertListEqual(data.tolist(), data2.tolist())

        self.assertEqual(len(result), 2)
        for i, r in enumerate(result):
            self.assertEqual(r[0], SplittingOperations.GET_NEXT_OP)
            self.assertIn('data', r[1])
            self.assertEqual(parent.children[0], r[1]['parent'])
            self.assertEqual(r[1]['pos'], i)
            self.assertListEqual(scope, r[1]['scope'])
            self.assertEqual(r[1]['data'].shape[1], data.shape[1])
            self.assertEqual(r[1]['data'].shape[0], int(np.sum(split_idx == i)))

        self.assertListEqual(result[0][1]['data'].tolist(), data[split_idx == 0, :].tolist())
        self.assertListEqual(result[1][1]['data'].tolist(), data[split_idx == 1, :].tolist())
        self.assertAlmostEqual(np.sum(parent.children[0].weights), 1.0)

    def test_create_sum_without_split(self):
        np.random.seed(17)
        data = np.arange(0, 1000).reshape(-1, 8)

        parent = Sum()
        parent.children.append(None)

        ctx = Context()
        ctx.feature_size = 4

        scope = [1, 3, 4, 6]
        data2 = np.array(data)

        K = int(data.shape[0] * 0.25)
        split_idx = np.array([0] * K + [1] * (data.shape[0] - K))
        np.random.shuffle(split_idx)

        def split_rows(data, context, scope):
            result = []
            result.append((data[split_idx == 0, :], scope, 0.25))
            result.append((data[split_idx == 1, :], scope, 0.75))
            return result

        result = create_sum(data=data2, parent=parent, pos=0, context=ctx, scope=list(scope),
                            split_rows=split_rows, split_on_sum=False)
        self.assertListEqual(data.tolist(), data2.tolist())

        self.assertEqual(len(result), 2)
        for i, r in enumerate(result):
            self.assertEqual(r[0], SplittingOperations.GET_NEXT_OP)
            self.assertIn('data', r[1])
            self.assertEqual(parent.children[0], r[1]['parent'])
            self.assertEqual(r[1]['pos'], i)
            self.assertListEqual(scope, r[1]['scope'])
            self.assertEqual(r[1]['data'].shape[1], data.shape[1])
            self.assertEqual(r[1]['data'].shape[0], data.shape[0])

        self.assertListEqual(result[0][1]['data'].tolist(), data.tolist())
        self.assertListEqual(result[1][1]['data'].tolist(), data.tolist())
        self.assertAlmostEqual(np.sum(parent.children[0].weights), 1.0)
        self.assertListEqual(parent.children[0].weights, [0.25, 0.75])

    def test_create_conditional(self):

        np.random.seed(17)
        data = np.arange(0, 1000).reshape(-1, 8)

        parent = Sum()
        parent.children.append(None)

        ctx = Context()
        ctx.feature_size = 4

        scope = [1, 3, 4, 6]
        data2 = np.array(data)

        K = int(data.shape[0] * 0.25)
        split_idx = np.array([0] * K + [1] * (data.shape[0] - K))
        np.random.shuffle(split_idx)

        y, x = get_YX(data, 4)

        def label_conditional(local_y, local_x):
            self.assertListEqual(local_y.tolist(), y.tolist())
            self.assertListEqual(local_x.tolist(), x.tolist())
            return split_idx

        result = create_conditional(data=data2, parent=parent, pos=0, context=ctx, scope=list(scope),
                                    label_conditional=label_conditional)

        self.assertListEqual(data.tolist(), data2.tolist())

        self.assertEqual(len(result), 2)

        for i, r in enumerate(result):
            self.assertEqual(r[0], SplittingOperations.GET_NEXT_OP)
            self.assertIn('data', r[1])
            self.assertEqual(parent.children[0], r[1]['parent'])
            self.assertEqual(r[1]['pos'], i)
            self.assertListEqual(scope, r[1]['scope'])
            self.assertEqual(r[1]['data'].shape[1], data.shape[1])

        conditional_node = result[0][1]['parent']

        child_idx = conditional_supervised_likelihood(conditional_node,
                                                      [np.zeros((data.shape[0], 1)), np.ones((data.shape[0], 1))], data)

        self.assertListEqual(result[0][1]['data'].tolist(), data[child_idx[:, 0] == 0, :].tolist())
        self.assertListEqual(result[1][1]['data'].tolist(), data[child_idx[:, 0] == 1, :].tolist())


if __name__ == '__main__':
    unittest.main()
