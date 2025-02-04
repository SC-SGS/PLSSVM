#!/usr/bin/env python3
# -*- coding: utf-8 -*-

########################################################################################################################
# Authors: Alexander Van Craen, Marcel Breyer                                                                          #
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved                                                   #
# License: This file is part of the PLSSVM project which is released under the MIT license.                            #
#          See the LICENSE.md file in the project root for full license information.                                   #
########################################################################################################################

import argparse
import re
import ctypes
from pathlib import Path
import subprocess

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--quiet", help="only output the final PLSSVM_TARGET_PLATFORMS string", action="store_true")
parser.add_argument("--gpus_only", help="only output gpu architectures to the final PLSSVM_TARGET_PLATFORMS string",
                    action="store_true")
args = parser.parse_args()


def cond_print(msg=""):
    if not args.quiet:
        print(msg)


# construct PLSSVM_TARGET_PLATFORMS string
plssvm_target_platforms = []
if not args.gpus_only:
    # CPU SIMD information for cpu target
    simd_version_support = {
        "avx512": False,
        "avx2":   False,
        "avx":    False,
        "sse4_2": False,
    }

    proc = subprocess.Popen(["lscpu"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    for simd_version in simd_version_support.keys():
        if simd_version in str(out):
            simd_version_support[simd_version] = True

    cond_print("supported CPU SIMD flags: {}\n".format(simd_version_support))

    newest_simd_version = ""
    for key in simd_version_support:
        if simd_version_support[key]:
            newest_simd_version = key
            break

    newest_simd_version = newest_simd_version.replace("_", ".")
    plssvm_target_platforms.append("cpu" + ("" if "".__eq__(newest_simd_version) else ":") + newest_simd_version)

# NVIDIA GPU information
nvidia_gpus = []
try:
    cuda_driver = ctypes.cdll.LoadLibrary("libcuda.so")
    cuda_flags = ctypes.c_int(0)

    cuda_driver.cuInit(cuda_flags)
    cuda_count = ctypes.c_int(0)
    cuda_driver.cuDeviceGetCount(ctypes.pointer(cuda_count))

    for device in range(cuda_count.value):
        major = ctypes.c_int(0)
        minor = ctypes.c_int(0)
        cuda_driver.cuDeviceComputeCapability(ctypes.pointer(major), ctypes.pointer(minor), device)
        target = "sm_{}{}".format(major.value, minor.value)
        nvidia_gpus.append(target)

    if len(nvidia_gpus)>0:
        cond_print("Found {} NVIDIA GPU(s): [{}]\n".format(len(nvidia_gpus), ", ".join(nvidia_gpus)))
        plssvm_target_platforms.append("nvidia:" + ",".join(set(nvidia_gpus)))
except:
    pass

# AMD GPU information
amd_gpus = []
amd_nodes = Path('/sys/class/kfd/kfd/topology/nodes')
# check if the nodes directory exists
if amd_nodes.is_dir():
    label = 'gfx_target_version '
    # iterate over all property files
    for filename in amd_nodes.glob('*/properties'):
        with filename.open('r') as prop:
            # iterate over all lines in the property file
            for line in prop:
                # check if the current line is the correct one
                if not line.startswith(label):
                    continue
                # convert the version to an integer
                version = int(line[len(label):])
                if not version:
                    break
                # convert the version to a gfx string
                major_version = version // 10000
                minor_version = (version // 100) % 100
                step_version = version % 100
                target = 'gfx{:d}{:x}{:x}'.format(major_version, minor_version, step_version)
                amd_gpus.append(target)

if len(amd_gpus)>0:
    cond_print("Found {} AMD GPU(s): [{}]\n".format(len(amd_gpus), ", ".join(amd_gpus)))
    plssvm_target_platforms.append("amd:" + ",".join(set(amd_gpus)))

# Intel GPU information
intel_gpus = []
pci_entry = subprocess.Popen(["lspci -nn | grep  -Ei 'VGA|DISPLAY'"], stdout=subprocess.PIPE, shell=True)
(out, err) = pci_entry.communicate()
for vga in str(out).splitlines():
    # check if the device is an Intel GPU
    if "Intel" in vga:
        # extract the architecture hex-value from the lspci line
        regex_pattern = r"\[[0-9]+:(.*?)\]"
        pci_value = re.search("\[[0-9]+:(.*?)\]", vga)
        if pci_value:
            value = pci_value.group(1)
            intel_gpus.append("0x{}".format(value))

if len(intel_gpus)>0:
    cond_print("Found {} Intel GPU(s): [{}]\n".format(len(intel_gpus), ", ".join(intel_gpus)))
    plssvm_target_platforms.append("intel:" + ",".join(set(intel_gpus)))

cond_print("Possible -DPLSSVM_TARGET_PLATFORMS entries:")
print(";".join(plssvm_target_platforms))
