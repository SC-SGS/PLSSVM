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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--file", help="the input file", required=True)
parser.add_argument("--col", help="", required=True)
parser.add_argument("--row", help="", required=True)
parser.add_argument("--val", help="", required=True)

args = parser.parse_args()

try:
    data = pd.read_csv(args.file)
except FileNotFoundError:
    data = pd.DataFrame()

data.at[int(args.row), args.col] = args.val
data.to_csv(args.file, index=False)
