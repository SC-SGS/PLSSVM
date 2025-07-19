# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn.datasets import make_circles, make_classification, make_moons, make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import sklearn
import plssvm

# create all classifiers
classifiers = [
    ("Linear SVM (ovr)", sklearn.svm.SVC(kernel="linear", C=0.025, random_state=42, decision_function_shape='ovr')),
    ("Linear SVM (ovo)", sklearn.svm.SVC(kernel="linear", C=0.025, random_state=42, decision_function_shape='ovo')),
    ("RBF SVM (ovr)", sklearn.svm.SVC(gamma=2, C=1, random_state=42, decision_function_shape='ovr')),
    ("RBF SVM (ovo)", sklearn.svm.SVC(gamma=2, C=1, random_state=42, decision_function_shape='ovo')),
    ("PLSSVM Linear (ovr)", plssvm.svm.SVC(kernel="linear", decision_function_shape='ovr', C=0.025)),
    ("PLSSVM Linear (ovo)", plssvm.svm.SVC(kernel="linear", decision_function_shape='ovo', C=0.025)),
    ("PLSSVM RBF (ovr)", plssvm.svm.SVC(gamma=2, decision_function_shape='ovr', C=1)),
    ("PLSSVM RBF (ovo)", plssvm.svm.SVC(gamma=2, decision_function_shape='ovo', C=1)),
    ("Neural Net", MLPClassifier(alpha=1, max_iter=1000, random_state=42)),
]

# create all datasets
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [
    ("moons", make_moons(noise=0.3, random_state=0)),
    ("circles", make_circles(noise=0.2, factor=0.5, random_state=1)),
    ("linear", linearly_separable),
    ("blobs", make_blobs(n_features=2, centers=4, random_state=0))
]

# plot everything
figure = plt.figure(figsize=(27, 9))
i = 1
# iterate over datasets
for ds_cnt, (ds_name, ds) in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    # just plot the dataset first
    if len(np.unique(y)) == 2:
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(["#FF0000", "#0000FF"])
    else:
        cm = plt.cm.tab10
        cm_bright = ListedColormap(["tab:blue", "tab:orange", "tab:green", "tab:red"])

    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
    # plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="k")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    print("{}:".format(ds_name))
    best_model = None
    best_model_score = 0.0
    # iterate over classifiers
    for name, clf in classifiers:
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        # fit the data
        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print("{}: {}".format(name, score))
        if score>best_model_score:
            best_model_score = score
            best_model = name

        # create the decision boundary
        DecisionBoundaryDisplay.from_estimator(clf, X, cmap=cm, alpha=0.8, ax=ax, eps=0.5)

        # plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="k")
        # plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="k", alpha=0.6)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name)

        # add the model accuracy as textbox
        ax.text(x_max - 0.3, y_min + 0.3, ("%.2f" % score).lstrip("0"), size=15,
                bbox=dict(boxstyle="round", alpha=0.8, facecolor="white"),
                horizontalalignment="right")
        i += 1
    print("Best model: {} ({})".format(best_model, best_model_score))
    print()

plt.tight_layout()
plt.show()
