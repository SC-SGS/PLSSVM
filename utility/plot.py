#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author Alexander Van Craen
@author Marcel Breyer
@copyright 2018-today The PLSSVM project - All Rights Reserved
@license This file is part of the PLSSVM project which is released under the MIT license.
         See the LICENSE.md file in the project root for full license information.
"""

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.ticker as mticker
# TODO: in eigenes Repo !!


def p2f(x):
    return [float(i.strip('%'))/100 for i in x]


data = pd.read_csv(
    "https://ipvs.informatik.uni-stuttgart.de/cloud/s/pYcggBo9bAJjb9B/download")
x = data.loc[(data["e"] == 0.0001), ["num_data_points"]]
y = data.loc[(data["e"] == 0.0001), ["total_time"]]

data["backend"] = data["backend"].fillna(data["svm"])


i = 1
num_plots = len(data["e"].unique())
for e in data["e"].unique():
    fig = plt.figure()
    # ax = fig.add_subplot(np.ceil(np.sqrt(num_plots)),np.ceil(np.sqrt(num_plots)),i,projection='3d')
    ax = fig.add_subplot(projection='3d')
    i = i + 1

    for backend in data.loc[(data["e"] == e), "backend"].unique():
        # ax.scatter(data.loc[(data["backend"] == backend) & (data["e"] == e) , "num_data_points"], data.loc[(data["backend"] == backend) & (data["e"] == e) ,"num_features"], data.loc[(data["backend"] == backend) & (data["e"] == e) ,"total_time"], label=backend)
        # ax.scatter(np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e) , "num_data_points"]), np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e) ,"num_features"]), np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e) ,"total_time"]), label=backend)
        # ax.scatter(np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e) , "num_data_points"]), np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e) ,"num_features"]), p2f(data.loc[(data["backend"] == backend) & (data["e"] == e) ,"accuracy"]), label=backend)
        surf = ax.plot_trisurf(np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e), "num_data_points"]), np.log10(data.loc[(data["backend"] == backend) & (
            data["e"] == e), "num_features"]), np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e), "total_time"]), label=backend, linewidth=0.0, antialiased=True,  alpha=0.5)
        surf._facecolors2d = surf._facecolor3d
        surf._edgecolors2d = surf._edgecolor3d

    def log_tick_formatter(val, pos=None):
        return r"$10^{:.0f}$".format(val)

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
    # ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

    ax.set_xlabel("num_data_points")
    ax.set_ylabel("num_features")
    ax.set_zlabel("total_time")
    ax.set_title("precision e=" + str(e))
    plt.legend()
    plt.show()


# i = 1;
# num_plots = len(data["e"].unique())
# for backend in data["backend"].unique():
#     fig = plt.figure()
#     # ax = fig.add_subplot(np.ceil(np.sqrt(num_plots)),np.ceil(np.sqrt(num_plots)),i,projection='3d')
#     ax = fig.add_subplot(projection='3d')
#     i = i + 1

#     for e in  data.loc[ (data["backend"] == backend) , "e"].unique():
#         surf = ax.plot_trisurf(np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e)  , "num_data_points"]), np.log10(data.loc[(data["backend"] == backend) & (data["e"] == e) ,"num_features"]), p2f(data.loc[(data["backend"] == backend) & (data["e"] == e) ,"accuracy"]), label=e, linewidth=0.0, antialiased=True,  alpha=0.5)
#         surf._facecolors2d = surf._facecolor3d
#         surf._edgecolors2d = surf._edgecolor3d


#     def log_tick_formatter(val, pos=None):
#         return r"$10^{:.0f}$".format(val)

#     ax.xaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
#     ax.yaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
#     # ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))

#     ax.set_xlabel("num_data_points")
#     ax.set_ylabel("num_features")
#     ax.set_zlabel("accuracy")
#     ax.set_title("backend=" + backend)
#     plt.legend()
#     plt.show()


# plt.show()
