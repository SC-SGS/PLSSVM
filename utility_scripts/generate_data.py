#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################################################################
# Authors: Alexander Van Craen, Marcel Breyer                                                                          #
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved                                                   #
# License: This file is part of the PLSSVM project which is released under the MIT license.                            #
#          See the LICENSE.md file in the project root for full license information.                                   #
########################################################################################################################

import argparse
import math
from timeit import default_timer as timer
import os
import humanize
import datetime
import importlib.util

# data set creation
from sklearn.datasets import make_classification
from sklearn.datasets import make_regression
from sklearn.datasets import make_friedman1
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles
from sklearn.preprocessing import minmax_scale

has_plssvm_python_bindings = importlib.util.find_spec("plssvm") is not None


# custom help action to print help for subparsers
class _SubParserHelpAction(argparse._HelpAction):

    def __call__(self, parser, namespace, values, option_string=None):
        # print main help
        parser.print_help()

        # retrieve subparsers from parser
        subparsers_actions = [action for action in parser._actions if isinstance(action, argparse._SubParsersAction)]
        for subparsers_action in subparsers_actions:
            # get all subparsers and print help
            for choice, subparser in subparsers_action.choices.items():
                print("\n\n{} specific arguments:\n".format(choice))
                subparser.print_help()

        parser.exit()


# parse command line arguments
parser = argparse.ArgumentParser(add_help=False)
subparsers = parser.add_subparsers(dest="task")

parser.add_argument(
    "-h", "-?", "--help", help="show this help message and exit",
    action=_SubParserHelpAction)
parser.add_argument(
    "--output", help="the output file to write the samples to (without extension)")
parser.add_argument(
    "--format", help="the file format; either arff, libsvm, or csv",
    default="libsvm")
parser.add_argument(
    "--samples", help="the number of training samples to generate",
    required=True, type=int)
parser.add_argument(
    "--test_samples", help="the number of test samples to generate; default: 0",
    type=int, default=0)
parser.add_argument(
    "--features", help="the number of features per data point",
    required=True, type=int)
parser.add_argument(
    "--scale", help="scale the features to the provided range",
    nargs=2, type=float)
parser.add_argument(
    "--plot", help="plot training samples; only possible if 0 < samples <= 2000 and 1 < features <= 3",
    action="store_true")

# create classification specific parser
classification_parser = subparsers.add_parser("classification", help="create a classification data set")
classification_parser.add_argument(
    "--problem", help="the problem to solve",
    choices=["blobs", "blobs_merged", "planes", "ball"], default="blobs")
classification_parser.add_argument(
    "--classes", help="the number of classes to generate; default: 2",
    type=int, default=2)

# create regression specific parser
regression_parser = subparsers.add_parser("regression", help="create regression data set")
regression_parser.add_argument(
    "--problem", help="the problem to solve",
    choices=["linear", "linear_noisy", "friedman1"], default="linear")

args = parser.parse_args()

# check for valid command line arguments
if args.samples<=0 or args.test_samples<0 or args.features<=0:
    raise RuntimeError(
        "Number of samples and/or features cannot be 0 or negative!")
if args.plot and (args.samples>2000 and (args.features != 2 or args.features != 3)):
    raise RuntimeError(
        "Invalid number of samples and/or features for plotting!")
if args.scale:
    if len(args.scale) != 2:
        raise RuntimeError(
            "Invalid number of arguments for scaling (must be exactly two)!")
    if args.scale[0]>=args.scale[1]:
        raise RuntimeError(
            "Invalid scaling range (first value must be smaller than the second value)!")

# set total number of samples
num_samples = args.samples + args.test_samples

print("Start creating {} data set samples... ".format(args.task), end="", flush=True)
start_time = timer()
# create labeled data set
if args.task == "classification":
    if args.problem == "blobs":
        samples, labels = make_blobs(
            n_samples=num_samples, n_features=args.features, centers=args.classes)
    elif args.problem == "blobs_merged":
        samples, labels = make_blobs(
            n_samples=num_samples, n_features=args.features, centers=args.classes, cluster_std=4.0)
    elif args.problem == "planes":
        samples, labels = make_classification(n_samples=num_samples, n_features=args.features,
                                              n_informative=math.ceil(math.sqrt(args.classes)),
                                              n_clusters_per_class=1, n_classes=args.classes)
    elif args.problem == "ball":
        samples, labels = make_gaussian_quantiles(
            n_samples=num_samples, n_features=args.features, n_classes=args.classes)
    else:
        raise RuntimeError("Invalid problem!")
elif args.task == "regression":
    if args.problem == "linear":
        samples, labels = make_regression(n_samples=num_samples, n_features=args.features)
    elif args.problem == "linear_noisy":
        samples, labels = make_regression(n_samples=num_samples, n_features=args.features, noise=30)
    elif args.problem == "friedman1":
        samples, labels = make_friedman1(n_samples=num_samples, n_features=args.features)
    else:
        raise RuntimeError("Invalid problem!")
else:
    raise RuntimeError("Invalid type!")

end_time = timer()
print("Done in {}ms.".format(int((end_time - start_time) * 1000)))

# if requested, scale the data
if args.scale:
    print("Scaling the data set to the range [{}, {}]... ".format(args.scale[0], args.scale[1]), end="", flush=True)
    start_time = timer()
    minmax_scale(samples, feature_range=(args.scale[0], args.scale[1]), copy=False)
    end_time = timer()
    print("Done in {}ms.".format(int((end_time - start_time) * 1000)))

print("Saving samples using {} ... ".format(
    "plssvm::data_set::save" if has_plssvm_python_bindings else "sklearn.datasets.dump_svmlight_file"), end="",
    flush=True)
start_time = timer()
# set file names
if args.output is not None:
    rawfile = os.path.join(args.output, "{}x{}".format(args.samples, args.features)) if os.path.isdir(
        args.output) else args.output
else:
    rawfile = "{}x{}".format(args.samples, args.features)

if rawfile.endswith(args.format):
    rawfile = rawfile[:-(len(args.format) + 1)]
file = rawfile + "." + args.format
test_file = ""
if args.test_samples>0:
    test_file = rawfile + "_test." + args.format

# save the files
if args.format == "libsvm":
    if has_plssvm_python_bindings:
        # save the libsvm file using the "fast" PLSSVM function
        import plssvm

        plssvm.set_verbosity(plssvm.VerbosityLevel.QUIET)

        # dump data in libsvm format
        if args.task == "classification":
            data_set = plssvm.ClassificationDataSet(samples[:args.samples, :], labels[:args.samples])
            data_set.save(file, format=plssvm.FileFormatType.LIBSVM)
            if args.test_samples>0:
                test_data_set = plssvm.ClassificationDataSet(samples[args.samples:, :], labels[args.samples:])
                test_data_set.save(file, format=plssvm.FileFormatType.LIBSVM)
        elif args.task == "regression":
            data_set = plssvm.RegressionDataSet(samples[:args.samples, :], labels[:args.samples])
            data_set.save(file, format=plssvm.FileFormatType.LIBSVM)
            if args.test_samples>0:
                test_data_set = plssvm.RegressionDataSet(samples[args.samples:, :], labels[args.samples:])
                test_data_set.save(file, format=plssvm.FileFormatType.LIBSVM)
        else:
            raise RuntimeError("Invalid type!")
    else:
        # save the libsvm file using the "slow" sklearn function
        from sklearn.datasets import dump_svmlight_file

        # dump data in libsvm format
        dump_svmlight_file(samples[:args.samples, :],
                           labels[:args.samples],
                           file,
                           zero_based=False)
        if args.test_samples>0:
            dump_svmlight_file(samples[args.samples:, :],
                               labels[args.samples:],
                               test_file,
                               zero_based=False)
elif args.format == "arff":
    import numpy
    import arff
    import pandas

    # dump data in arff format
    # concatenate features and labels
    data = numpy.c_[samples, labels]

    # convert numpy array to pandas dataframe
    col_names = ["feature_" + str(i) for i in range(args.features)]
    col_names.append("class")


    def dump_arff_file(out_data, out_file, relation):
        # dump dataframe as arff file
        pd_data = pandas.DataFrame(data=out_data, columns=col_names)
        arff.dump(out_file, pd_data.values,
                  relation=relation, names=pd_data.columns)

        # replace 'real' with 'numeric' in arff file
        with open(file) as f:
            new_text = f.read().replace('real', 'numeric')
            new_text = new_text.replace('class numeric', 'class {-1,1}')
            new_text = "% This data set has been created at {}\n% {}x{}\n".format(
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), args.samples, args.features) + new_text
        with open(file, "w") as f:
            f.write(new_text)


    # dump arff files
    dump_arff_file(data[:args.samples, :], file, '\"train data set\"')
    if args.test_samples>0:
        dump_arff_file(data[args.samples:, :], test_file, '\"test data set\"')
elif args.format == "csv":
    import numpy

    # concatenate features and labels of the training data set
    data = numpy.c_[samples, labels]
    numpy.savetxt(file, data[:args.samples, :], delimiter=',')
    # concatenate features and labels of the test data set if necessary
    if args.test_samples>0:
        data = numpy.c_[samples, labels]
        numpy.savetxt(test_file, data[args.samples:, :], delimiter=',')
else:
    raise RuntimeError("Only arff, libsvm, and csv supported as file formats!")

end_time = timer()
print("Done in {}ms.".format(int((end_time - start_time) * 1000)))

# output info
if args.task == "classification":
    print(
        "Created a classification training data set '{}' ({}) with {} data points, {} features, and {} classes.".format(
            file, humanize.naturalsize(os.path.getsize(file)), args.samples, args.features, args.classes))
    if args.test_samples>0:
        print("Created a classification test data set '{}' ({}) with {} data points, {} features, and {} classes."
              .format(test_file, humanize.naturalsize(os.path.getsize(file)), args.test_samples, args.features,
                      args.classes))
elif args.task == "regression":
    print("Created a regression training data set '{}' ({}) with {} data points and {} features.".format(
        file, humanize.naturalsize(os.path.getsize(file)), args.samples, args.features))
    if args.test_samples>0:
        print("Created a regression test data set '{}' ({}) with {} data points and {} features."
              .format(test_file, humanize.naturalsize(os.path.getsize(file)), args.test_samples, args.features))
else:
    raise RuntimeError("Invalid type!")

# plot generated data set
if args.plot:
    # plotting imports
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    if args.task == "classification":
        if args.features == 2:
            plt.scatter(samples[:args.samples, 0],
                        samples[:args.samples, 1], c=labels)
        elif args.features == 3:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(samples[:args.samples, 0], samples[:args.samples, 1], samples[:args.samples, 2], c=labels)
    elif args.task == "regression":
        if args.features == 1:
            plt.scatter(samples[:args.samples, 0], labels[:args.samples])
        elif args.features == 2:
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.scatter(samples[:args.samples, 0], samples[:args.samples, 1], labels[:args.samples])
    else:
        raise RuntimeError("Invalid type!")
    plt.show()
