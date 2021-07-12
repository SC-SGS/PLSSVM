#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Alexander Van Craen
@author Marcel Breyer
"""

import argparse
import numpy as np
import pandas
import arff

# data set creation
from sklearn.datasets import dump_svmlight_file
from sklearn.datasets import make_classification
from sklearn.datasets import make_blobs
from sklearn.datasets import make_gaussian_quantiles

# plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--output", help="the output file to write the samples to (without extension)", required=True)
parser.add_argument("--format", help="the file format; either arff or libsvm", required=True)
parser.add_argument("--problem", help="the problem to solve; one of: blobs, blobs_merged, planes, planes_merged, ball",
                    default="blobs")
parser.add_argument("--samples", help="the number of training samples to generate", required=True, type=int)
parser.add_argument("--test_samples", help="the number of test samples to generate; default: 0", type=int, default=0)
parser.add_argument("--features", help="the number of features per data point", required=True, type=int)
parser.add_argument("--plot", help="plot training samples; only possible if 0 < samples <= 2000 and 1 < features <= 3",
                    action="store_true")

args = parser.parse_args()

# check for valid command line arguments
if args.samples <= 0 or args.test_samples < 0 or args.features <= 0:
    raise RuntimeError("Number of samples and/or features cannot be 0 or negative!")
if args.plot and (args.samples > 2000 and (args.features != 2 or args.features != 3)):
    raise RuntimeError("Invalid number of samples and/or features for plotting!")

# set total number of samples
num_samples = args.samples + args.test_samples

# create labeled data set # TODO: do they make sense?
if args.problem == "blobs":
    samples, labels = make_blobs(n_samples=num_samples, n_features=args.features, centers=2)
elif args.problem == "blobs_merged":
    samples, labels = make_blobs(n_samples=num_samples, n_features=args.features, centers=2, cluster_std=4.0)
elif args.problem == "planes":
    samples, labels = make_classification(n_samples=num_samples, n_features=args.features, n_redundant=0,
                                          n_informative=2, n_clusters_per_class=1)
elif args.problem == "planes_merged":
    samples, labels = make_classification(n_samples=num_samples, n_features=args.features, n_redundant=0,
                                          n_informative=args.features)
elif args.problem == "ball":
    samples, labels = make_gaussian_quantiles(n_samples=num_samples, n_features=args.features, n_classes=2)
else:
    raise RuntimeError("Invalid problem!")

# map labels to -1 and 1
labels = labels * 2 - 1


# set file names
file = args.output + "." + args.format
if args.test_samples > 0:
    test_file = args.output + "_test." + args.format

if args.format == "libsvm":
    # dump data in libsvm format
    dump_svmlight_file(samples[:args.samples, :], labels[:args.samples], file)
    if args.test_samples > 0:
        dump_svmlight_file(samples[args.samples:, :], labels[args.samples:], test_file)
elif args.format == "arff":
    # dump data in arff format
    # concatenate features and labels
    data = np.c_[samples, labels]

    # convert numpy array to pandas dataframe
    col_names = ["feature_" + str(i) for i in range(args.features)]
    col_names.append("class")

    def dump_arff_file(out_data, out_file, relation):
        # dump dataframe as arff file
        pd_data = pandas.DataFrame(data=out_data, columns=col_names)
        arff.dump(out_file, pd_data.values, relation=relation, names=pd_data.columns)

        # replace 'real' with 'numeric' in arff file
        with open(file) as f:
            new_text = f.read().replace('real', 'numeric')
        with open(file, "w") as f:
            f.write(new_text)

    # dump arff files
    dump_arff_file(data[:args.samples, :], file, 'train data set')
    if args.test_samples > 0:
        dump_arff_file(data[args.samples:, :], test_file, 'test data set')

else:
    raise RuntimeError("Only arff and libsvm supported as file format!")


# output info
print("Created training data set '{}' with {} data points and {} features.".format(file, args.samples, args.features))
if args.test_samples > 0:
    print("Created test data set '{}' with {} data points and {} features."
          .format(test_file, args.test_samples, args.features))

# plot generated data set
if args.plot:
    if args.features == 2:
        plt.scatter(samples[:args.samples, 0], samples[:args.samples, 1], c=labels)
    elif args.features == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(samples[:args.samples, 0], samples[:args.samples, 1], samples[:args.samples, 2], c=labels)
    plt.show()
