#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Alexander Van Craen
@author Marcel Breyer
@copyright 2018-today The PLSSVM project - All Rights Reserved
@license This file is part of the PLSSVM project which is released under the MIT license.
         See the LICENSE.md file in the project root for full license information.
"""

import re
import pandas as pd
import os
import sys
import argparse
from datetime import datetime
import platform

# set up regular expressions
rx_dict = {
    # always present, if print_info is set to true (-q;--quiet is not present on the command line)
    'backend': re.compile(r'Using (?P<backend>\S+) as backend.'),
    'num_gpus': re.compile(r'Found (?P<num_gpus>\d+) \S+ device\(s\):'),
    'num_data_points': re.compile(r'(?P<num_data_points>\d+) data points'),
    'num_features': re.compile(r'(?P<num_features>\d+) features'),
    'time_read': re.compile(r'features in (?P<time_read>\d+)ms using'),
    'file_parser': re.compile(r'(?P<file_parser>\S+) parser'),
    'time_transform': re.compile(r'Transformed dataset from 2D AoS to 1D SoA in (?P<time_transform>\d+)ms'),
    'time_setup': re.compile(r'Setup for solving the optimization problem done in (?P<time_setup>\d+)ms.'),
    'max_iterations': re.compile(r'Iteration \d+ \(max: (?P<max_iterations>\d+)\)'),
    'iterations': re.compile(r'Finished after (?P<iterations>\d+)'),
    'time_learn': re.compile(r'using CG in (?P<time_learn>\d+)ms'),
    'kernel_type': re.compile(r'kernel_type (?P<kernel_type>\S+)'),
    'rho': re.compile(r'rho (?P<rho>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'),
    'time_write': re.compile(r'Wrote model file with \d+ support vectors in (?P<time_write>\d+)ms.'),
    # additional information if the plssvm::parameter struct ist outputted
    'degree': re.compile(r'degree *(?P<degree>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'),
    'gamma': re.compile(r'gamma *(?P<gamma>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'),
    'coef0': re.compile(r'coef0 *(?P<coef0>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'),
    'cost': re.compile(r'cost *(?P<cost>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'),
    'epsilon': re.compile(r'epsilon *(?P<epsilon>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'),
    'real_type': re.compile(r'real_type *(?P<epsilon>\S+)')
}


def _parse_line(line):
    """
    Do a regex search against all defined regexes and
    return a list of key and match results

    """
    results = []
    for key, rx in rx_dict.items():
        match = rx.search(line)
        if match:
            results.append((key, match))
    return results


def parse_stream(file_stream):
    """
    Parse text on a given file_stream

    Parameters
    ----------
    file_stream : str
        file_object to be parsed

    Returns
    -------
    data : pd.DataFrame
        Parsed data

    """
    now = datetime.now()
    # create an dictionary to collect the data
    data = {
        "parse_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "system": platform.node(),
    }
    for line in file_stream:
        # at each line check for a match with a regex
        matches = _parse_line(line)
        for key, match in matches:
            for val in match.groupdict():
                # check if match exists
                if match.group(val) is not None:
                    # check if value already read
                    if key in data:
                        if data[key] != match.group(val):
                            raise Exception(
                                file_stream, 'ill-formed in line:', line)
                    else:
                        # add new value to dictionary
                        data[key] = match.group(val)

    return pd.DataFrame.from_records([data])


if __name__ == '__main__':
    stdin = False

    # use stdin if it's full
    if not sys.stdin.isatty():
        input_stream = sys.stdin
        stdin = True

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="the input file", required=not stdin)
    parser.add_argument(
        "--output", help="the output file to write the parsed values to", required=True)

    args = parser.parse_args()

    # read the given filename even if stdin is full
    if args.input:
        input_stream = open(args.input, 'r')

    # parse
    data = parse_stream(input_stream)

    try:
        # if outfile already exists concatenate
        if os.path.isfile(args.output):
            data = pd.concat([data, pd.read_csv(args.output)])
    except pd.errors.EmptyDataError:
        pass

    # write results
    data.to_csv(args.output, index=False)
