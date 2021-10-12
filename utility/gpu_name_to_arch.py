#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@author Alexander Van Craen
@author Marcel Breyer
@copyright 2018-today The PLSSVM project - All Rights Reserved
@license This file is part of the PLSSVM project which is released under the MIT license.
         See the LICENSE.md file in the project root for full license information.
"""

import argparse

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--name", help="the full name of the GPU (e.g. GeForce RTX 3080)")
args = parser.parse_args()

if args.name is None:
    # for nvidia GPUs
    import GPUtil
    # for AMD GPUs
    import pyamdgpuinfo

    gpu_names = []
    # check for possible NVIDIA GPUs
    gpu_names.extend([gpu.name for gpu in GPUtil.getGPUs()])
    # check for possible AMD GPUs
    gpu_names.extend([pyamdgpuinfo.get_gpu(
        gpu_id).name for gpu_id in range(pyamdgpuinfo.detect_gpus())])
    if not gpu_names:
        # error if no GPUs where found
        raise RuntimeError("Couldn't find any NVIDIA or AMD GPU(s)!")
    else:
        print("Found {} GPU(s):\n{}".format(len(gpu_names), gpu_names))
else:
    # use provided GPU name
    gpu_names = [args.name]

# mapping of NVIDIA compute capabilities given the GPU name
# only GPUs with a compute capability greater or equal than 6.0 are support
# https://developer.nvidia.com/cuda-gpus
nvidia_compute_capability_mapping = {
    # Datacenter Products
    "NVIDIA A100": "sm_80",
    "NVIDIA A40": "sm_86",
    "NVIDIA A30": "sm_80",
    "NVIDIA A10": "sm_86",
    "NVIDIA A16": "sm_86",
    "NVIDIA T4": "sm_75",
    "NVIDIA V100": "sm_70",
    "Tesla P100": "sm_60",
    "Tesla P40": "sm_61",
    "Tesla P4": "sm_61",
    # NVIDIA Quadro and NVIDIA RTX
    "RTX A6000": "sm_86",
    "RTX A5000": "sm_86",
    "RTX A4000": "sm_86",
    "T1000": "sm_75",
    "T600": "sm_75",
    "T400": "sm_75",
    "Quadro RTX 8000": "sm_75",
    "Quadro RTX 6000": "sm_75",
    "Quadro RTX 5000": "sm_75",
    "Quadro RTX 4000": "sm_75",
    "Quadro GV100": "sm_70",
    "Quadro GP100": "sm_60",
    "Quadro P6000": "sm_61",
    "Quadro P5000": "sm_61",
    "Quadro P4000": "sm_61",
    "Quadro P2200": "sm_61",
    "Quadro P2000": "sm_61",
    "Quadro P1000": "sm_61",
    "Quadro P620": "sm_61",
    "Quadro P600": "sm_61",
    "Quadro P400": "sm_61",
    "RTX A3000": "sm_86",
    "RTX A2000": "sm_86",
    "RTX 5000": "sm_75",
    "RTX 4000": "sm_75",
    "RTX 3000": "sm_75",
    "T2000": "sm_75",
    "T1200": "sm_75",
    "T500": "sm_75",
    "P620": "sm_61",
    "P520": "sm_61",
    "Quadro P5200": "sm_61",
    "Quadro P4200": "sm_61",
    "Quadro P3200": "sm_61",
    "Quadro P3000": "sm_61",
    "Quadro P500": "sm_61",
    # GeForce and TITAN Products
    "GeForce RTX 3060 Ti": "sm_86",
    "GeForce RTX 3060": "sm_86",
    "GeForce RTX 3090": "sm_86",
    "GeForce RTX 3080": "sm_86",
    "GeForce RTX 3070": "sm_86",
    "GeForce GTX 1650 Ti": "sm_75",
    "NVIDIA TITAN RTX": "sm_75",
    "GeForce RTX 2080 Ti": "sm_75",
    "GeForce RTX 2080": "sm_75",
    "GeForce RTX 2070": "sm_75",
    "GeForce RTX 2060": "sm_75",
    "NVIDIA TITAN V": "sm_70",
    "NVIDIA TITAN Xp": "sm_61",
    "NVIDIA TITAN X": "sm_61",
    "GeForce GTX 1080 Ti": "sm_61",
    "GeForce GTX 1080": "sm_61",
    "GeForce GTX 1070 Ti": "sm_61",
    "GeForce GTX 1070": "sm_61",
    "GeForce GTX 1060": "sm_61",
    "GeForce GTX 1050": "sm_61",
    "GeForce RTX 3050 Ti": "sm_86",
    "GeForce RTX 3050": "sm_86",
    # Jetson Products
    "Jetson AGX Xavier": "sm_72",
}

# mapping of AMD architectures given the GPU name
# https://github.com/RadeonOpenCompute/ROCm_Documentation/blob/master/ROCm_Compiler_SDK/ROCm-Native-ISA.rst#id145
amd_arch_mapping = {
    "Radeon VII": "gfx906",
    "Radeon Instinct MI50": "gfx906",
    "Radeon Instinct MI6": "gfx906",
    "Ryzen 3 2200G": "gfx902",
    "Ryzen 5 2400G": "gfx902",
    "Radeon Vega Frontier Edition": "gfx900",
    "Radeon RX Vega 56": "gfx900",
    "Radeon RX Vega 64": "gfx900",
    "Radeon RX Vega 64 Liquid Cooled": "gfx900",
    "Radeon Instinct MI25": "gfx900",
    "Radeon RX 460": "gfx803",
    "Radeon RX 470": "gfx803",
    "Radeon RX 480": "gfx803",
    "Radeon R9 Nano": "gfx803",
    "Radeon R9 Fury": "gfx803",
    "Radeon R9 FuryX": "gfx803",
    "Radeon Pro Duo FirePro S9300x2": "gfx803",
    "Radeon Instinct MI8": "gfx803",
}

# output mapped name
for name in gpu_names:
    if name in nvidia_compute_capability_mapping:
        print(nvidia_compute_capability_mapping[name])
    elif name in amd_arch_mapping:
        name = name.replace("AMD", "").strip()
        print(amd_arch_mapping[name])
    else:
        raise RuntimeError("Unrecognized GPU name {}".format(name))
