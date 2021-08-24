#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

data.at[int(args.row),args.col] = args.val
data.to_csv(args.file, index=False)
