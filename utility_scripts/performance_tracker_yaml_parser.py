#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################################################################
# Authors: Alexander Van Craen, Marcel Breyer                                                                          #
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved                                                   #
# License: This file is part of the PLSSVM project which is released under the MIT license.                            #
#          See the LICENSE.md file in the project root for full license information.                                   #
########################################################################################################################

import argparse
import matplotlib.pyplot as plt

# parse the YAML file
import yaml

# parse the runtimes given as strings, e.g., '125ms'
from pint import UnitRegistry

ureg = UnitRegistry()
ureg.setup_matplotlib(True)

# parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("--tracking_file", help="the YAML file storing the tracked performance", required=True)

args = parser.parse_args()

# read the yaml file
with open(args.tracking_file) as file:
    data = list(yaml.safe_load_all(file))

    # plot the total runtimes of training tasks based on the number of data points
    # NOTE: this example ignores the number of features for an easier plot

    training_data = [d for d in data if d["parameter"]["task"] == "train"]
    x = [d["data_set_read"]["num_data_points"] for d in training_data]
    y = [ureg.Quantity(d["total_time"]) for d in training_data]

    # plot the data using matplotlib
    plt.scatter(x, y)
    plt.show()
