import json
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_similarity_score,
    roc_auc_score,
    make_scorer,
    log_loss,
)

import numpy as np
from sklearn.multioutput import ClassifierChain

from experiments.multilabel_classification import get_dataset
from structure.Conditional.Inference import add_conditional_inference_support

np.random.seed(1)

from ScikitCSPNClassifier import CSPNClassifier

scorer = make_scorer(accuracy_score, greater_is_better=True)

path = os.path.abspath(os.path.dirname(__file__))
spn_folder = path + "/spn_cache/"


class naive_model:
    def fit(self, X, y):
        self.num_X = X.shape[1]
        self.num_Y = y.shape[1]
        return self

    def predict(self, X):
        return np.zeros((X.shape[0], self.num_Y))


def micro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="micro")


def train_baseline(ds_name, train_input, train_labels):

    tuned_params = {"random_state": [i for i in np.arange(10)]}
    base_lr = LogisticRegression(
        C=1, max_iter=500, fit_intercept=True, tol=1e-15, class_weight="balanced"
    )
    gs_chain = GridSearchCV(
        ClassifierChain(base_lr, order="random"), tuned_params, cv=3, scoring=scorer
    )
    gs_chain.fit(train_input, train_labels)

    print("best order according to grid search is %s" % gs_chain.best_estimator_.order_)
    print("best order according to grid search is %s" % gs_chain.best_score_)

    from sklearn.ensemble import RandomForestClassifier

    tuned_params = {
        "n_estimators": [i for i in np.arange(1, 100, 10)],
        "max_depth": [i for i in np.arange(1, 100, 10)],
    }
    gs_forest = GridSearchCV(
        RandomForestClassifier(random_state=1), tuned_params, cv=3, scoring=scorer
    )
    gs_forest.fit(train_input, train_labels)

    return (
        ("classifier_chain", gs_chain.best_estimator_),
        ("random_forest", gs_forest.best_estimator_),
    )


def get_ds_property(ds_name, train_input, train_labels, test_input):
    base_rates = np.sum(train_labels, axis=0) / len(train_labels)
    global ds_prop
    ds_prop = {}
    base_rate = {}
    ds_prop["ds_name"] = ds_name
    ds_prop["num_train_instance"] = len(train_input)
    ds_prop["num_test_instance"] = len(test_input)
    ds_prop["num_features"] = train_input.shape[1]
    ds_prop["num_labels"] = train_labels.shape[1]
    ds_prop["base_rate"] = base_rates.tolist()
    print(json.dumps(ds_prop, indent=4))


if __name__ == "__main__":
    add_conditional_inference_support()

    from sklearn.model_selection import GridSearchCV

    ds_name = "emotions"
    X_train, X_test, y_train, y_test = get_dataset(ds_name)
    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    num_X = X_train.shape[1]
    num_Y = y_train.shape[1]

    min_instances_slice = np.arange(10, len(X_train), 20).tolist()
    alpha = [10 ** (-i) for i in range(1, 5)]
    tuned_params = {"min_instances_slice": min_instances_slice, "alpha": alpha}

    print(
        CSPNClassifier(cluster_univariate=False, allow_sum_nodes=True)
        .get_params()
        .keys()
    )
    gs_ = GridSearchCV(
        CSPNClassifier(cluster_univariate=False, allow_sum_nodes=True),
        tuned_params,
        cv=3,
        scoring=scorer,
    )
    gs_.fit(X_train, y_train)
    print(gs_.best_score_)

    # for some reason I have to pass y with same shape
    # otherwise gridsearch throws an error. Not sure why.
    best_parameters = gs_.best_params_
    cspn = gs_.best_estimator_

    fname = spn_folder + ds_name + "/" + "cspn.bin"
    pickle.dump(cspn, open(fname, "wb"))
    print("best parameters is %s" % best_parameters)

    models = [("cspn", cspn)]
    baselines = train_baseline(ds_name, X_train, y_train)
    models.extend(baselines)
    models.append(("naive", naive_model().fit(X_train, y_train)))
    models.append(
        (
            "naive_cspn",
            CSPNClassifier(cluster_univariate=False, allow_sum_nodes=True, min_instances_slice=100000000).fit(
                X_train, y_train
            ),
        )
    )

    chain, forest = train_baseline(ds_name, X_train, y_train)

    get_ds_property(ds_name, X_train, y_train, X_test)

    average = True
    results = {}
    for idx, (mname, model) in enumerate(models):
        model_results = {}
        results[mname] = model_results
        y_pred = model.predict(X_test)
        try:
            model_results["accuracy"] = accuracy_score(y_pred, y_test, normalize=True)
        except:
            print("%s has problem in accuracy" % mname)
        try:
            model_results["auc"] = roc_auc_score(y_test, model.predict_proba(X_test))
        except:
            print("%s has problem in auc" % mname)
        try:
            model_results["f1 macro"] = f1_score(y_pred, y_test, average="macro")
            model_results["f1 micro"] = f1_score(y_pred, y_test, average="micro")
        except:
            print("%s has problem in f1" % mname)

    print(json.dumps(results, sort_keys=False, indent=4))
