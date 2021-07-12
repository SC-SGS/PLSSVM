#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author Alexander Van Craen
@author Marcel Breyer
"""

import re
import pandas as pd
import os
import sys
import argparse


# set up regular expressions
rx_dict = {
    'iterations': re.compile(r'Start Iteration:? (?P<iteration>\d+)'),
    'rho': re.compile(r'rho (?P<rho>[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'),
    'time_read': re.compile(r'(?P<time>\d+) ms eingelesen'),
    'time_load': re.compile(r'(?P<time>\d+) ms auf die Gpu geladen'),
    'time_learn': re.compile(r'(?P<time>\d+) ms gelernt'),
    'time_write': re.compile(r'(?P<time>\d+) ms geschrieben'),
    'points': re.compile(r'(?P<points>\d+) (Datenpunkte)|(data points)'),
    'features': re.compile(r'(Dimension (?P<dimension>\d+))|((?P<dimension2>\d+) features)'),
    'num_gpus': re.compile(r'GPUs found: (?P<num_gpus>\d+)'),
    'transform': re.compile(r'(?P<time>\d+) ms transformiert'),

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
    data = {}  # create an empty dictionary to collect the data
    for line in file_stream:
        # at each line check for a match with a regex
        matches = _parse_line(line)
        for key, match in matches:
            for val in match.groupdict():
                # check if match exists
                if match.group(val) != None:
                    # check if value already read
                    if key in data:
                        # update iteration number
                        if key == 'iterations':
                            data[key] = max(data[key], match.group(val))
                        # throw exception if value is already read differently
                        elif data[key] != match.group(val):
                            raise Exception(
                                file_stream, 'illformed in line:', line)
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
    parser.add_argument("--input", help="the input file", required= not stdin)
    parser.add_argument("--output", help="the output file to write the samples to (without extension)", required=True)

    args = parser.parse_args()

    # read the given filename even if stdin is full
    if args.input:
        input_stream = open(args.input, 'r')

    # parse
    data = parse_stream(input_stream)

    # if outfile already exists concatenate
    if os.path.isfile(args.output):
        data = pd.concat([data, pd.read_csv(args.output)])
    #write results
    data.to_csv(args.output, index=False)
