#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################################################################
# Authors: Alexander Van Craen, Marcel Breyer                                                                          #
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved                                                   #
# License: This file is part of the PLSSVM project which is released under the MIT license.                            #
#          See the LICENSE.md file in the project root for full license information.                                   #
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt

# generate sample data (sine curve with noise)
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel()

# add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

plt.scatter(X, y, color='darkorange', label='data')

# fit the sklearn regression model
from sklearn.svm import SVR

sklearn_svr_lin = SVR(kernel='linear', C=100, epsilon=0.1)
y_lin_sklearn = sklearn_svr_lin.fit(X, y).predict(X)
plt.plot(X, y_lin_sklearn, lw=2, linestyle='dashed', label='Linear model sklearn')

sklearn_svr_poly = SVR(kernel='poly', C=100, degree=3, epsilon=0.1, coef0=1)
y_poly_sklearn = sklearn_svr_poly.fit(X, y).predict(X)
plt.plot(X, y_poly_sklearn, lw=2, linestyle='dashed', label='Polynomial model sklearn')

sklearn_svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
y_rbf_sklearn = sklearn_svr_rbf.fit(X, y).predict(X)
plt.plot(X, y_rbf_sklearn, lw=2, linestyle='dashed', label='RBF model sklearn')

# fit the PLSSVM regression model
from plssvm.svm import SVR

plssvm_svr_lin = SVR(kernel='linear', C=100)
y_lin_plssvm = plssvm_svr_lin.fit(X, y).predict(X)
plt.plot(X, y_lin_plssvm, lw=2, label='Linear model plssvm')

plssvm_svr_poly = SVR(kernel='poly', C=100, degree=3, coef0=1)
y_poly_plssvm = plssvm_svr_poly.fit(X, y).predict(X)
plt.plot(X, y_poly_plssvm, lw=2, label='Polynomial model plssvm')

plssvm_svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1)
y_rbf_plssvm = plssvm_svr_rbf.fit(X, y).predict(X)
plt.plot(X, y_rbf_plssvm, lw=2, label='RBF model plssvm')

# show the result plots
plt.xlabel('data')
plt.ylabel('target')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
