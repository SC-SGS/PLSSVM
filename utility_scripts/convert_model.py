#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################################################################
# Authors: Alexander Van Craen, Marcel Breyer                                                                          #
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved                                                   #
# License: This file is part of the PLSSVM project which is released under the MIT license.                            #
#          See the LICENSE.md file in the project root for full license information.                                   #
########################################################################################################################

import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("model_file", help="the regression model file to convert")
parser.add_argument("-o", "--output", help="output the regression model to the new file, otherwise the regression model us updated inplace")
parser.add_argument("--to_plssvm", help="convert the regression model to a PLSSVM conform model file", action="store_true")
parser.add_argument("--to_libsvm", help="convert the regression model to a LIBSVM conform model file", action="store_true")
args = parser.parse_args()

# sanity check parameters
if (args.to_plssvm and args.to_libsvm) or (not args.to_plssvm and not args.to_libsvm):
    raise RuntimeError("One of --to_plssvm or --to_libsvm must be used! Either none or both were provided!")

# read the original model file
with open(args.model_file, 'r') as file:
    filedata = file.read()

# replace the target string depending on the model file to which the current model should be converted to
must_update_model_file = False
if "svm_type c_svr" not in filedata:
    if args.to_plssvm:
        must_update_model_file = True
        filedata = filedata.replace("svm_type epsilon_svr", "svm_type c_svr")
        print("Converting a LIBSVM regression model file to a PLSSVM regression model file.")
    else:
        print("Already a PLSSVM regression model file. Nothing to convert.")
elif "svm_type epsilon_svr" not in filedata:
    if args.to_libsvm:
        must_update_model_file = True
        filedata = filedata.replace("svm_type c_svr", "svm_type epsilon_svr")
        print("Converting a PLSSVM regression model file to a LIBSVM regression model file.")
    else:
        print("Already a LIBSVM regression model file. Nothing to convert.")
else:
    raise RuntimeError("Invalid regression model file!")

# check whether we work inplace or not
if args.output is None:
    # inplace -> check if the file must be updated or can stay as is
    if must_update_model_file:
        # write the new model to a new file
        with open(args.model_file, 'w') as file:
            file.write(filedata)
else:
    # write the new model to the same file (inplace)
    with open(args.output, 'w') as file:
        file.write(filedata)