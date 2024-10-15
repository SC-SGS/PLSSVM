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

import pylspci  # get Intel GPU information

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--quiet", help="only output the final PLSSVM_TARGET_PLATFORMS string", action="store_true")
parser.add_argument("--gpus_only", help="only output gpu architectures to the final PLSSVM_TARGET_PLATFORMS string",
                    action="store_true")
args = parser.parse_args()


def cond_print(msg=""):
    if not args.quiet:
        print(msg)


# TODO: maybe use "lspci -nn |grep  -Ei 'VGA|DISPLAY'" ?
# mapping of Intel architecture names
# https://dgpu-docs.intel.com/devices/hardware-table.html
# https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html
intel_arch_mapping = {
    # Skylake
    "skl":     ["192A", "1932", "193B", "193A", "193D", "1923", "1926", "1927", "192B", "192D", "1912", "191B", "1913",
                "1915", "1917", "191A", "1916", "1921", "191D", "191E", "1902", "1906", "190B", "190A", "190E"],
    # Gemini Lake
    "glk":     ["3185", "3184"],
    # Apollo Lake
    "Gen9":    ["1A85", "5A85", "0A84", "1A84", "5A84"],
    # Kaby Lake
    "kbl":     ["593B", "5923", "5926", "5927", "5917", "5912", "591B", "5916", "5921", "591A", "591D", "591E", "591C",
                "87C0", "5913", "5915", "5902", "5906", "590B", "590A", "5908", "590E"],
    # Coffee Lake
    "cfl":     ["3EA5", "3EA8", "3EA6", "3EA7", "3EA2", "3E90", "3E93", "3E99", "3E9C", "3EA1", "9BA5", "9BA8", "3EA4",
                "9B21", "9BA0", "9BA2", "9BA4", "9BAA", "9BAB", "9BAC", "87CA", "3EA3", "9B41", "9BC0", "9BC2", "9BC4",
                "9BCA", "9BCB", "9BCC", "3E91", "3E92", "3E98", "3E9B", "9BC5", "9BC8", "3E96", "3E9A", "3E94", "9BC6",
                "9BE6", "9BF6", "3EA9", "3EA0"],
    # Ice Lake
    "icllp":   ["8A71", "8A56", "8A58", "8A5B", "8A5D", "8A54", "8A5A", "8A5C", "8A57", "8A59", "8A50", "8A51", "8A52",
                "8A53"],
    # Tiger Lake
    "tgllp":   ["9A60", "9A68", "9A70", "9A40", "9A49", "9A78", "9AC0", "9AC9", "9AD9", "9AF8"],
    # Xe MAX
    "dg1":     ["4905"],
    # Rocket Lake
    "Gen11":   ["4C8A", "4C8B", "4C90", "4C9A", "4C8C"],
    #
    "Gen12LP": [],
    # Alder Lake
    "adls":    ["4680", "4682", "4688", "468A", "4690", "4692", "4693"],
    # Alder Lake
    "aldp":    ["4626", "4628", "462A", "46A0", "46A1", "46A2", "46A3", "46A6", "46A8", "46AA", "46B0", "46B1", "46B2",
                "46B3", "46C0", "46C1", "46C2", "46C3", "46D0", "46D1", "46D2"],
    # # Alchemist
    # "": ["5690", "56A0", "5691", "56A1", "5692", "56A5" ,"5693", "5694", "56A6"],
    # # Flex Series
    # "": ["56C1", "56C0"],
}
intel_arch_to_name_mapping = {
    "skl":     "Skylake with Intel Processor Graphics Gen9",
    "kbl":     "Kaby Lake with Intel Processor Graphics Gen9",
    "cfl":     "Coffee Lake with Intel Processor Graphics Gen9",
    "glk":     "Gemini Lake with Intel Processor Graphics Gen9",
    "icllp":   "Ice Lake with Intel Processor Graphics Gen11",
    "tgllp":   "Tiger Lake with Intel Processor Graphics Gen12",
    "dg1":     "Intel Iris Xe MAX graphics",
    "Gen9":    "Intel Processor Graphics Gen9",
    "Gen11":   "Rocket Lake with Intel Processor Graphics Gen11",
    "Gen12LP": "Intel Processor Graphics Gen12 (Lower Power)",
    "adls":    "Alder Lake S with Intel Processor Graphics Gen12.2",
    "aldp":    "Alder Lake P with Intel Processor Graphics Gen12.2",
}

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
intel_gpu_names = []
for device in pylspci.parsers.SimpleParser().run():
    if re.search("VGA", str(device.cls), re.IGNORECASE):
        for key in intel_arch_mapping:
            if any(re.search(arch, str(device.device), re.IGNORECASE) for arch in intel_arch_mapping[key]):
                intel_gpu_names.append(str(device.device))
                break
intel_num_gpus = len(intel_gpu_names)

if intel_num_gpus>0:
    intel_gpus = { x: intel_gpu_names.count(x) for x in intel_gpu_names }
    intel_gpu_arch = { }
    for name in intel_gpus:
        for key in intel_arch_mapping:
            if any(re.search(arch, name, re.IGNORECASE) for arch in intel_arch_mapping[key]):
                intel_gpu_arch[name] = key
                break

    cond_print("Found {} Intel (i)GPU(s):".format(intel_num_gpus))
    for name in intel_gpus:
        cond_print("  {}x {} ({}): {}".format(intel_gpus[name], name,
                                              intel_arch_to_name_mapping[intel_gpu_arch[name]],
                                              intel_gpu_arch[name]))
    cond_print()

    plssvm_target_platforms.append("intel:" + ",".join({ str(sm) for sm in intel_gpu_arch.values() }))

cond_print("Possible -DPLSSVM_TARGET_PLATFORMS entries:")
print(";".join(plssvm_target_platforms))
