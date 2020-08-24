import sys
sys.path.append('../')
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from spn.algorithms.Inference import likelihood
from spn.structure.Base import Context
from spn.structure.leaves.parametric.Parametric import Bernoulli

from algorithms.ConditionalStructureLearning import create_sum
from algorithms.ExactMPE import ExactMPE
from algorithms.splitting.Clustering import get_split_conditional_rows_KMeans
from structure.Conditional.Inference import add_conditional_inference_support
from structure.Conditional.Sampling import add_conditional_sampling_support
from structure.Conditional.ScikitLinearModel import CSPNLinearModel
from structure.Conditional.Supervised import create_conditional_leaf
from structure.Conditional.utils import concatenate_yx

print(__doc__)

# Code source: Gaël Varoquaux
#              Andreas Müller
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

add_conditional_sampling_support()
add_conditional_inference_support()


class CSPNTestClassifier(BaseEstimator):

    def fit(self, X, y=None):
        y = y.reshape(y.shape[0], -1)
        self.num_labels = y.shape[1]
        self.context = Context(parametric_types=[Bernoulli] * self.num_labels).add_domains(y)
        self.context.feature_size = X.shape[1]
        self.scope = list(range(y.shape[1]))
        data = concatenate_yx(y, X)


        cspn_type = 1
        if cspn_type == 0:
            self.cspn = create_conditional_leaf(data, self.context, self.scope)
        elif cspn_type == 1:
            split_rows = get_split_conditional_rows_KMeans()
            self.cspn, subtasks = create_sum(data=data, node_id=0, parent_id=0, pos=0, context=self.context,
                                         scope=self.scope, split_rows=split_rows)
            for i, subtask in enumerate(subtasks):
                self.cspn.children[i] = create_conditional_leaf(subtask[1]['data'], self.context, subtask[1]['scope'])
            print(self.cspn)

    def predict_proba(self, X):
        y = np.ones((X.shape[0], self.num_labels))
        y[:] = np.nan

        test_data = concatenate_yx(y, X)

        results = np.ones_like(y)

        for n in self.cspn.scope:
            local_test = np.array(test_data)
            local_test[:, n] = 1
            results[:, n] = likelihood(self.cspn, local_test)[:, 0]

        rbinc = np.zeros((X.shape[0], 2))
        rbinc[:, 0] = 1 - results[:, 0]
        rbinc[:, 1] = results[:, 0]
        return rbinc

    def predict(self, X, check_input=True):
        if self.cspn is None:
            raise RuntimeError("Classifier not fitted")

        y = np.array([np.nan] * X.shape[0] * len(self.cspn.scope)).reshape(X.shape[0], -1)

        test_data = concatenate_yx(y, X)

        mpe_y = ExactMPE(self.cspn, test_data, self.context)

        return mpe_y

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)


names = ["CSPN", "Bagging","Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    CSPNTestClassifier(),
    BaggingClassifier(base_estimator=GaussianProcessClassifier(1.0 * RBF(1.0)),n_estimators=3, random_state=17, max_samples=0.2),
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=2),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    ]

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k', alpha=0.6)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()
plt.show()
