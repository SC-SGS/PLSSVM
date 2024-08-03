#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################################################################
# Authors: Alexander Van Craen, Marcel Breyer                                                                          #
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved                                                   #
# License: This file is part of the PLSSVM project which is released under the MIT license.                            #
#          See the LICENSE.md file in the project root for full license information.                                   #
########################################################################################################################

import argparse

from sklearn.datasets import make_classification
from wrapt_timeout_decorator import *
import math
from datetime import datetime
from timeit import default_timer as timer

import plssvm


def all_same(items):
    return len(set(items)) < 2


class CGTimeout(Exception):
    """ custom timeout exception """


@timeout(600, timeout_exception=CGTimeout)
def fit_model_with_timeout(csvm, data, eps):
    return csvm.fit(data, epsilon=eps)


# parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--num_data_points", help="the number of data points used for this performance measurements", required=True,
    type=int)
parser.add_argument(
    "--num_features", help="the number of features per data point used for this performance measurements",
    required=True, type=int)
parser.add_argument(
    "--num_repeats", help="the number of repeats for this performance measurements", required=True, type=int)
parser.add_argument(
    "--performance_tracking", help="the output YAML file where the performance tracking results are written to",
    default="tracking.yaml")
parser.add_argument(
    "--intermediate_train_file",
    help="the name of the intermediate training data file used to also track IO timings; can also be used to store the file for further tests",
    default="train_data.libsvm")
parser.add_argument(
    "--intermediate_model_file",
    help="the name of the intermediate model file used to also track IO timings; can also be used to store the model for further tests",
    default="train_data.libsvm.model")

args = parser.parse_args()

print("Generating data set {}x{} using sklearn".format(args.num_data_points, args.num_features))
samples, labels = make_classification(n_samples=args.num_data_points, n_features=args.num_features, n_redundant=0,
                                      n_informative=2, n_clusters_per_class=1)

try:
    params = plssvm.Parameter()

    plssvm.detail.PerformanceTracker.pause()

    # create the data set
    train_data = plssvm.DataSet(samples, labels, scaling=(-1.0, 1.0))
    train_data.save(args.intermediate_train_file, plssvm.FileFormatType.LIBSVM)

    # create a C-SVM using the provided parameters and the default, i.e., fastest backend and target platform
    svm = plssvm.CSVM(params)

    plssvm.quiet()

    # find an epsilon such that the resulting model has an accuracy of over 97%
    print("Determining the used epsilon value")
    epsilon = 0.0
    accuracies = []
    for eps_exp in range(1, 21):
        epsilon = pow(10, -eps_exp)
        # fit using the training data
        model = fit_model_with_timeout(svm, train_data, epsilon)

        # get accuracy of the trained model
        accuracies.append(svm.score(model) * 100.0)

        print("accuracy {:.2f} for epsilon {}".format(accuracies[-1], epsilon))
        # if the current accuracy is greater than 97% stop
        if accuracies[-1] > 97:
            break
        # if the accuracy is greater than 70% and the last three accuracies where the same,
        # use the smallest epsilon and break
        if accuracies[-1] > 70 and len(accuracies) >= 3 and all_same(accuracies[-3:]):
            epsilon = pow(10, -(accuracies.index(max(accuracies)) + 1))
            break

    # plssvm.set_verbosity(plssvm.VerbosityLevel.FULL)
    plssvm.detail.PerformanceTracker.resume()
    achieved_accuracy = accuracies[int(abs(math.log(epsilon, 10))) - 1]
    print("Using {} as epsilon value for an accuracy of {:.2f} %".format(epsilon, achieved_accuracy))

    # create a list of all available backends
    print("Performing the actual performance measurements")
    available_backends = []
    for backend in plssvm.list_available_backends():
        # skip the automatic type
        if backend == plssvm.BackendType.AUTOMATIC:
            continue

        if backend == plssvm.BackendType.SYCL:
            # special case SYCL backend
            # add all available SYCL implementation and both kernel invocation types
            available_sycl_implementations = plssvm.sycl.list_available_sycl_implementations()
            available_sycl_implementations.reverse()
            for sycl_impl in available_sycl_implementations:
                # skip the automatic type
                if sycl_impl == plssvm.sycl.ImplementationType.AUTOMATIC:
                    continue
                available_backends.append((backend, {"sycl_implementation_type": sycl_impl,
                                                     "sycl_kernel_invocation_type": plssvm.sycl.KernelInvocationType.ND_RANGE}))
        else:
            available_backends.append((backend, {}))

    # generate runtimes for all available backends
    for backend in available_backends:
        print("\nGenerating runtimes for the {}".format(backend[0]))
        if backend[0] == plssvm.BackendType.SYCL:
            print("SYCL: {}".format(backend[1]))

        for rep in range(args.num_repeats):
            print("{} repetition {}".format(datetime.now(), rep))

            start_time = timer()

            # create the data set
            train_data = plssvm.DataSet(args.intermediate_train_file)

            # create a default C-SVM
            svm = plssvm.CSVM(backend[0], **backend[1])

            # use the found value of epsilon to collect runtimes for an SVM model with an accuracy of more than 97%
            try:
                model = fit_model_with_timeout(svm, train_data, epsilon)
            except CGTimeout:
                print("timeout occurred! skipping remaining repeats for the {}".format(backend[0]))
                break

            # save the model file
            model.save(args.intermediate_model_file)

            end_time = timer()

            # save the performance tracking results
            plssvm.detail.PerformanceTracker.add_parameter_tracking_entry(params)
            plssvm.detail.PerformanceTracker.add_string_tracking_entry("", "accuracy", str(achieved_accuracy))
            plssvm.detail.PerformanceTracker.add_string_tracking_entry("", "total_runtime", "{}ms".format(
                int((end_time - start_time) * 1000)))
            plssvm.detail.PerformanceTracker.save(args.performance_tracking)

except plssvm.PLSSVMError as e:
    print(e)
except RuntimeError as e:
    print(e)
