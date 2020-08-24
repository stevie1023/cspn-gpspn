"""
Created on November 25, 2018

@author: Alejandro Molina
"""
import os
from pathlib import Path
from time import sleep

from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import ClassifierChain
from sklearn.svm import SVC
from skmultilearn.dataset import load_dataset

import numpy as np
from spn.algorithms.Inference import likelihood
from spn.io.plot import TreeVisualization
from spn.io.plot.TreeVisualization import set_symbol
from spn.structure.leaves.parametric.Parametric import Bernoulli

from RClassifier import RClassifier
from ScikitCSPNClassifier import CSPNClassifier
from data.multilabel.Fimp import Fimp
from structure.Conditional.Inference import add_conditional_inference_support
from structure.Conditional.Supervised import SupervisedOr
from structure.Conditional.utils import concatenate_yx


def get_dataset(dataset, ranked_features=None, reduce_dim=None, num_features=None):
    # Load a multi-label dataset from https://www.openml.org/d/40597
    # X, Y = fetch_mldata('yeast', version=4, return_X_y=True)
    if dataset == "yeast":
        data = fetch_mldata("yeast")
        X = data["data"]
        Y = data["target"].transpose().toarray()

        train_input, test_input, train_labels, test_labels = train_test_split(
            X, Y, test_size=0.2, random_state=0
        )

        if reduce_dim is not None and reduce_dim < train_input.shape[1]:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=reduce_dim)
            pca.fit(train_input)
            train_input = pca.transform(train_input)

            pca.fit(test_input)
            test_input = pca.transform(test_input)

        return train_input, test_input, train_labels, test_labels
    elif dataset == "emotions":
        train_input, train_labels, feature_names, label_names = load_dataset(
            "emotions", "train"
        )
        test_input, test_labels, _, _ = load_dataset("emotions", "test")
        fimp = Fimp(
            f_name=str(Path(os.path.dirname(__file__)).parent)
                   + "/data/multilabel/emotions.fimp"
        )
        indices = np.asarray(fimp.get_attr_indices())[range(-1, 2)] - 1

        return (
            train_input.toarray(),
            test_input.toarray(),
            train_labels.toarray(),
            test_labels.toarray(),
        )

    elif ds_name == "xor":
        dataIn, dataOut = get_xor()
        return train_test_split(dataIn, dataOut, random_state=17)
    elif ds_name == "moons":
        dataIn, dataOut = make_moons(
            n_samples=20000, shuffle=True, noise=0.1, random_state=17
        )
        dataOut = np.vstack((dataOut, dataOut, dataOut)).T
        return train_test_split(dataIn, dataOut, random_state=17)
    elif ds_name == "sys_multilabel":
        dataIn, dataOut = get_sy_multilabel()
        return train_test_split(dataIn, dataOut, random_state=17)
    else:
        train_input, train_labels, test_input, test_labels = get_categorical_data(
            dataset
        )
        train_input = np.asarray(train_input, dtype=np.float)
        train_labels = np.asarray(train_labels, dtype=np.int)
        test_input = np.asarray(test_input, dtype=np.float)
        test_labels = np.asarray(test_labels, dtype=np.int)

        return train_input, test_input, train_labels, test_labels


def eval_model(model, input_features, input_labels):
    pred_labels = model.predict(input_features)

    f1 = f1_score(input_labels, pred_labels, average="macro")

    pred_proba = model.predict_proba(input_features)

    if isinstance(model, RandomForestClassifier):
        if input_labels.shape[1] == 1:
            pred_proba = pred_proba[:, 1]
        else:
            pred_proba = np.array(pred_proba).swapaxes(0, 1)[:, :, 1]

    if isinstance(model, SVC):
        pred_proba = pred_proba[:, 1]

    auc = roc_auc_score(input_labels, pred_proba, average="weighted")
    return auc, f1


if __name__ == "__main__":
    add_conditional_inference_support()
    set_symbol(SupervisedOr, "⋏")

    ds_name = "emotions"

    print(ds_name)
    train_features, test_features, train_labels, test_labels = get_dataset(
        ds_name, num_features=3
    )

    # train_labels = train_labels[:, [4, 5]]
    # test_labels = test_labels[:, [4, 5]]

    models = []

    models.append((
        "rdn_forest",
        RandomForestClassifier(n_estimators=1, max_depth=50, random_state=1),
    )
    )

    ctree = RClassifier('ctree', packages=['partykit'])
    models.append(('ctree', ctree))

    naive_model = CSPNClassifier(parametric_types=[Bernoulli] * train_labels.shape[1],
                                 cluster_univariate=True, min_instances_slice=100000000, allow_sum_nodes=False
                                 )
    models.append(("naive", naive_model))

    deep_model = CSPNClassifier(parametric_types=[Bernoulli] * train_labels.shape[1],
                                cluster_univariate=True, min_instances_slice=2, allow_sum_nodes=False
                                )
    models.append(("deep", deep_model))

    models.append(
        (
            "chain",
            ClassifierChain(
                LogisticRegression(C=1, max_iter=300, fit_intercept=True, tol=1e-5),
                order="random",
                random_state=17,
            ),
        )
    )

    models.append(("svm", SVC(C=0.1, probability=True, class_weight="balanced")))

    results = []
    print("\n\n\n\n")
    for name, model in models:
        try:
            model.fit(train_features, train_labels)
            results.append(
                name
                + " train auc=%s, f1=%s"
                % eval_model(model, train_features, train_labels)
            )
            results.append(
                name
                + " test auc=%s, f1=%s" % eval_model(model, test_features, test_labels)
            )
            results.append(
                "-------------------------------------------------------------"
            )
        except:
            results.append("problem with model: " + name)

    print("\n".join(results))

    TreeVisualization.plot_spn(deep_model.cspn, file_name="/tmp/tree_deep_cspn.png")
    TreeVisualization.plot_spn(naive_model.cspn, file_name="/tmp/tree_naive_cspn.png")
