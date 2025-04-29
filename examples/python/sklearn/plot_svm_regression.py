"""
===================================================================
Support Vector Regression (SVR) using linear and non-linear kernels
===================================================================

Toy example of 1D regression using linear, polynomial and RBF kernels.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from plssvm.svm import SVR


def one_div_x(data):
    return np.asarray([1 / x for x in data])


def irregular_function(data):
    return np.asarray([
        0.1 * x ** 5 - 0.5 * x ** 3  # Polynomial base
        + 2 * np.exp(-0.2 * (x - 2) ** 2)  # Gaussian bump on the right
        - 3 * np.exp(-0.5 * (x + 1) ** 2)  # Sharp dip on the left
        + 1 / (x ** 2 + 1)  # Rational term to create non-linearity
        for x in data
    ])


# %%
# Generate sample data
# --------------------
data_sets = []
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()
# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))
data_sets.append((X, y))

X = np.sort(np.random.uniform(-2, 2, 40).reshape((40, 1)), axis=0)
y = one_div_x(X).ravel()
# add noise to targets
y[::5] += 5 * (0.5 - np.random.rand(8))
data_sets.append((X, y))

X = np.sort(np.random.uniform(-3, 3, 40).reshape((40, 1)), axis=0)
y = irregular_function(X).ravel()
# add noise to targets
y[::5] += 5 * (0.5 - np.random.rand(8))
data_sets.append((X, y))

# %%
# Fit regression model
# --------------------
svr_lin = SVR(kernel="linear", C=100, gamma="auto")
svr_poly = SVR(kernel="poly", C=100, gamma="auto", degree=3, coef0=1)
svr_rbf = SVR(kernel="rbf", C=100, gamma=0.1)
svr_sigmoid = SVR(kernel="sigmoid", C=100, gamma=0.1)
svr_laplacian = SVR(kernel="laplacian", C=100, gamma=0.1)

# %%
# Look at the results
# -------------------
svrs = [svr_lin, svr_poly, svr_rbf, svr_sigmoid, svr_laplacian]
kernel_label = ["Linear", "Polynomial", "RBF", "Sigmoid", "Laplacian"]
model_color = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

fig, axes = plt.subplots(nrows=3, ncols=len(svrs) + 1, figsize=(15, 8), sharey='row')

i = 0
for X, y in data_sets:
    if i == 0:
        print("sin:")
        values = np.linspace(np.min(X), np.max(X))
        axes[i][0].plot(values, np.sin(values), color='k', label="sin")
    elif i == 1:
        print("1/x:")
        values_lower = np.linspace(np.min(X), -0.001)
        values_upper = np.linspace(0.001, np.max(X))
        axes[i][0].plot(values_lower, one_div_x(values_lower), color='k', label="1/x")
        axes[i][0].plot(values_upper, one_div_x(values_upper), color='k')
        axes[i][0].set_ylim(-8, 8)
    else:
        print("irregular function:")
        values = np.linspace(np.min(X), np.max(X))
        axes[i][0].plot(values, irregular_function(values), color='k', label="irregular")
    axes[i][0].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=1,
        fancybox=True,
        shadow=True,
    )

    for ix, svr in enumerate(svrs):
        svr.fit(X, y)
        print("{}: {}".format(svr, svr.score(X, y)))

        axes[i][ix + 1].plot(
            X,
            svr.predict(X),
            color=model_color[ix],
            lw=2,
            label="{}".format(kernel_label[ix]),
        )
        axes[i][ix + 1].scatter(
            X[svr.support_],
            y[svr.support_],
            facecolor="none",
            edgecolor=model_color[ix],
            s=50,
        )
        axes[i][ix + 1].scatter(
            X[np.setdiff1d(np.arange(len(X)), svr.support_)],
            y[np.setdiff1d(np.arange(len(X)), svr.support_)],
            facecolor="none",
            edgecolor="k",
            s=50,
        )
        axes[i][ix + 1].legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=1,
            fancybox=True,
            shadow=True,
        )
    i = i + 1
    print()

fig.suptitle("Support Vector Regression", fontsize=14)
fig.tight_layout()
plt.show()
