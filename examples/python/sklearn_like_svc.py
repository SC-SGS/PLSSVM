#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################################################################
# Authors: Alexander Van Craen, Marcel Breyer                                                                          #
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved                                                   #
# License: This file is part of the PLSSVM project which is released under the MIT license.                            #
#          See the LICENSE.md file in the project root for full license information.                                   #
########################################################################################################################

import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.metrics
import sklearn.inspection
import numpy as np
from plssvm import SVC

# load the breast cancer datasets
cancer = sklearn.datasets.load_breast_cancer()
X = cancer.data[:, :2]
y = cancer.target
y_label = cancer.target_names

# build the SVC model
svm = SVC(kernel="rbf", gamma=0.5, C=1.0).fit(X, y)

# score the model
print(sklearn.metrics.classification_report(y, svm.predict(X)))
print("Score: {:.2f}%".format(svm.score(X, y) * 100))

# plot the decision boundary
sklearn.inspection.DecisionBoundaryDisplay.from_estimator(
    svm,
    X,
    response_method="predict",
    cmap=plt.cm.Spectral,
    alpha=0.8,
    xlabel=cancer.feature_names[0],
    ylabel=cancer.feature_names[1],
)

# scatter plot the decision boundary
viridis = plt.cm.get_cmap('viridis', len(np.unique(y)))
plt.scatter(X[:, 0], X[:, 1],
            cmap=viridis,
            c=y,
            s=20, edgecolors="k")

# generate legend handles and add handle
legend_handles = [plt.scatter([], [], color=viridis(color), label=f'{label}')
                  for label, color in zip(y_label, np.unique(y))]
plt.legend(handles=legend_handles)

plt.title("SVC classifier on breast cancer dataset")
plt.show()
