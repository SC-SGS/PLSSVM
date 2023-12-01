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

import cpuinfo  # get CPU SIMD information
import GPUtil  # get NVIDIA GPU information
import pyamdgpuinfo  # get AMD GPU information
import pylspci  # get Intel GPU information

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--quiet", help="only output the final PLSSVM_TARGET_PLATFORMS string", action="store_true")
args = parser.parse_args()


def cond_print(msg=""):
    if not args.quiet:
        print(msg)


# mapping of NVIDIA compute capabilities given the GPU name
# only GPUs with compute capability greater or equal than 6.0 are support
# https://developer.nvidia.com/cuda-gpus
nvidia_compute_capability_mapping = {
    # Datacenter Products
    "NVIDIA H100": "sm_90",
    "NVIDIA L4": "sm_89",
    "NVIDIA L40": "sm_89",
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
    "RTX 6000": "sm_89",
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
    "GeForce RTX 4090": "sm_89",
    "GeForce RTX 4080": "sm_89",
    "GeForce RTX 4070 Ti": "sm_89",
    "GeForce RTX 3060": "sm_86",
    "GeForce RTX 3090": "sm_86",
    "GeForce RTX 3090 Ti": "sm_86",
    "GeForce RTX 3080": "sm_86",
    "GeForce RTX 3080 Ti": "sm_86",
    "GeForce RTX 3070": "sm_86",
    "GeForce RTX 3070 Ti": "sm_86",
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
    "Jetson AGX Orin": "sm_87",
    "Jetson AGX Orin NX": "sm_87",
    "Jetson AGX Orin Nano": "sm_87",
    "Jetson AGX Xavier": "sm_72",
    "Jetson AGX Xavier NX": "sm_72",
}

# mapping of AMD architectures given the GPU name
# https://llvm.org/docs/AMDGPUUsage.html
amd_arch_mapping = {
    # AMD Radeon GPUs
    "AMD Instinct MI250X": "gfx90a",
    "AMD Instinct MI250": "gfx90a",
    "AMD Instinct MI210": "gfx90a",
    "Radeon RX 7900 XTX": "gfx1101",
    "Radeon RX 7900 XT": "gfx1101",
    "Radeon RX 7900 GRE": "gfx1101",
    "Radeon RX 7800 XT": "gfx1101",
    "Radeon RX 7700 XT": "gfx1101",
    "Radeon RX 7600": "gfx1101",
    "Radeon RX 6700 XT": "gfx1031",
    "Radeon RX 6800": "gfx1030",
    "Radeon RX 6800 XT": "gfx1030",
    "Radeon RX 6900 XT": "gfx1030",
    "Radeon RX 5500": "gfx1012",
    "Radeon RX 5500 XT": "gfx1012",
    "Radeon Pro V520": "gfx1011",
    "Radeon RX 5700": "gfx1010",
    "Radeon RX 5700 XT": "gfx1010",
    "Radeon Pro 5600 XT": "gfx1010",
    "Radeon Pro 5600M": "gfx1010",
    "Radeon Instinct MI100 Accelerator": "gfx908",
    "Radeon Pro VII": "gfx906",
    "Radeon VII": "gfx906",
    "Radeon Instinct MI50": "gfx906",
    "Radeon Instinct MI60": "gfx906",
    "Radeon Vega Frontier Edition": "gfx900",
    "Radeon RX Vega 56": "gfx900",
    "Radeon RX Vega 64": "gfx900",
    "Radeon RX Vega 64 Liquid": "gfx900",
    "Radeon Instinct MI25": "gfx900",
    "Radeon RX 460": "gfx803",
    "Radeon Instinct MI6": "gfx803",
    "Radeon RX 470": "gfx803",
    "Radeon RX 480": "gfx803",
    "Radeon Instinct MI8": "gfx803",
    "Radeon R9 Nano": "gfx803",
    "Radeon R9 Fury": "gfx803",
    "Radeon R9 FuryX": "gfx803",
    "Radeon Pro Duo": "gfx803",
    "Radeon R9 285": "gfx802",
    "Radeon R9 380": "gfx802",
    "Radeon R9 385": "gfx802",
    # AMD Ryzen iGPUs
    "Ryzen 7 4700G": "gfx90c",
    "Ryzen 7 4700GE": "gfx90c",
    "Ryzen 5 4600G": "gfx90c",
    "Ryzen 5 4600GE": "gfx90c",
    "Ryzen 3 4300G": "gfx90c",
    "Ryzen 3 4300GE": "gfx90c",
    "Ryzen Pro 4000G": "gfx90c",
    "Ryzen 7 Pro 4700G": "gfx90c",
    "Ryzen 7 Pro 4750GE": "gfx90c",
    "Ryzen 5 Pro 4650G": "gfx90c",
    "Ryzen 5 Pro 4650GE": "gfx90c",
    "Ryzen 3 Pro 4350G": "gfx90c",
    "Ryzen 3 Pro 4350GE": "gfx90c",
    "Ryzen 3 2200G": "gfx902",
    "Ryzen 5 2400G": "gfx902",
    # other AMD targets
    "FirePro S7150": "gfx805",
    "FirePro S7100": "gfx805",
    "FirePro W7100": "gfx805",
    "Mobile FirePro M7170": "gfx805",
    "FirePro S9300x2": "gfx803",
    "A6-8500P": "gfx801",
    "Pro A6-8500B": "gfx801",
    "A8-8600P": "gfx801",
    "Pro A8-8600B": "gfx801",
    "FX-8800P": "gfx801",
    "Pro A12-8800B": "gfx801",
    "A10-8700P": "gfx801",
    "Pro A10-8700B": "gfx801",
    "A10-8780P": "gfx801",
    "A10-9600P": "gfx801",
    "A12-9700P": "gfx801",
    "A12-9730P": "gfx801",
    "FX-9800P": "gfx801",
    "FX-9830P": "gfx801",
    "E2-9010": "gfx801",
    "A6-9210": "gfx801",
    "A9-9410": "gfx801",
}

# mapping of Intel architecture names
# https://dgpu-docs.intel.com/devices/hardware-table.html
# https://www.intel.com/content/www/us/en/develop/documentation/oneapi-dpcpp-cpp-compiler-dev-guide-and-reference/top/compilation/ahead-of-time-compilation.html
intel_arch_mapping = {
    # Skylake
    "skl": ["192A", "1932", "193B", "193A", "193D", "1923", "1926", "1927", "192B", "192D", "1912", "191B", "1913",
            "1915", "1917", "191A", "1916", "1921", "191D", "191E", "1902", "1906", "190B", "190A", "190E"],
    # Gemini Lake
    "glk": ["3185", "3184"],
    # Apollo Lake
    "Gen9": ["1A85", "5A85", "0A84", "1A84", "5A84"],
    # Kaby Lake
    "kbl": ["593B", "5923", "5926", "5927", "5917", "5912", "591B", "5916", "5921", "591A", "591D", "591E", "591C",
            "87C0", "5913", "5915", "5902", "5906", "590B", "590A", "5908", "590E"],
    # Coffee Lake
    "cfl": ["3EA5", "3EA8", "3EA6", "3EA7", "3EA2", "3E90", "3E93", "3E99", "3E9C", "3EA1", "9BA5", "9BA8", "3EA4",
            "9B21", "9BA0", "9BA2", "9BA4", "9BAA", "9BAB", "9BAC", "87CA", "3EA3", "9B41", "9BC0", "9BC2", "9BC4",
            "9BCA", "9BCB", "9BCC", "3E91", "3E92", "3E98", "3E9B", "9BC5", "9BC8", "3E96", "3E9A", "3E94", "9BC6",
            "9BE6", "9BF6", "3EA9", "3EA0"],
    # Ice Lake
    "icllp": ["8A71", "8A56", "8A58", "8A5B", "8A5D", "8A54", "8A5A", "8A5C", "8A57", "8A59", "8A50", "8A51", "8A52",
              "8A53"],
    # Tiger Lake
    "tgllp": ["9A60", "9A68", "9A70", "9A40", "9A49", "9A78", "9AC0", "9AC9", "9AD9", "9AF8"],
    # Xe MAX
    "dg1": ["4905"],
    # Rocket Lake
    "Gen11": ["4C8A", "4C8B", "4C90", "4C9A", "4C8C"],
    #
    "Gen12LP": [],
    # Alder Lake
    "adls": ["4680", "4682", "4688", "468A", "4690", "4692", "4693"],
    # Alder Lake
    "aldp": ["4626", "4628", "462A", "46A0", "46A1", "46A2", "46A3", "46A6", "46A8", "46AA", "46B0", "46B1", "46B2",
             "46B3", "46C0", "46C1", "46C2", "46C3", "46D0", "46D1", "46D2"],
    # # Alchemist
    # "": ["5690", "56A0", "5691", "56A1", "5692", "56A5" ,"5693", "5694", "56A6"],
    # # Flex Series
    # "": ["56C1", "56C0"],
}
intel_arch_to_name_mapping = {
    "skl": "Skylake with Intel Processor Graphics Gen9",
    "kbl": "Kaby Lake with Intel Processor Graphics Gen9",
    "cfl": "Coffee Lake with Intel Processor Graphics Gen9",
    "glk": "Gemini Lake with Intel Processor Graphics Gen9",
    "icllp": "Ice Lake with Intel Processor Graphics Gen11",
    "tgllp": "Tiger Lake with Intel Processor Graphics Gen12",
    "dg1": "Intel Iris Xe MAX graphics",
    "Gen9": "Intel Processor Graphics Gen9",
    "Gen11": "Rocket Lake with Intel Processor Graphics Gen11",
    "Gen12LP": "Intel Processor Graphics Gen12 (Lower Power)",
    "adls": "Alder Lake S with Intel Processor Graphics Gen12.2",
    "aldp": "Alder Lake P with Intel Processor Graphics Gen12.2",
}

# construct PLSSVM_TARGET_PLATFORMS string
plssvm_target_platforms = ""

# CPU SIMD information for cpu target
simd_version_support = {
    "avx512": False,
    "avx2": False,
    "avx": False,
    "sse4_2": False,
}

cpu_info = cpuinfo.get_cpu_info()

for flag in cpu_info["flags"]:
    for key in simd_version_support:
        if flag == key:
            simd_version_support[key] = True
    if flag.startswith("avx512"):
        simd_version_support["avx512"] = True

cond_print("{}: {}\n".format(cpu_info["brand_raw"], simd_version_support))

newest_simd_version = ""
for key in simd_version_support:
    if simd_version_support[key]:
        newest_simd_version = key
        break
plssvm_target_platforms += "cpu" + ("" if "".__eq__(newest_simd_version) else ":") + newest_simd_version

# NVIDIA GPU information
nvidia_gpu_names = [gpu.name for gpu in GPUtil.getGPUs()]
nvidia_num_gpus = len(nvidia_gpu_names)

if nvidia_num_gpus > 0:
    nvidia_gpus = {x: nvidia_gpu_names.count(x) for x in nvidia_gpu_names}
    nvidia_gpu_sm = {}
    # get NVIDIA SM from GPU name
    for name in nvidia_gpus:
        found_name = False
        for key in nvidia_compute_capability_mapping:
            if re.search(key, name, re.IGNORECASE):
                nvidia_gpu_sm[name] = nvidia_compute_capability_mapping[key]
                found_name = True
                break

        if not found_name:
            raise RuntimeError("Unrecognized GPU name '{}'".format(name))

    cond_print("Found {} NVIDIA GPU(s):".format(nvidia_num_gpus))
    for name in nvidia_gpus:
        cond_print("  {}x {}: {}".format(nvidia_gpus[name], name, nvidia_gpu_sm[name]))
    cond_print()

    plssvm_target_platforms += ";nvidia:" + ",".join({str(sm) for sm in nvidia_gpu_sm.values()})

# AMD GPU information
amd_gpu_names = [pyamdgpuinfo.get_gpu(gpu_id).name for gpu_id in range(pyamdgpuinfo.detect_gpus())]
amd_num_gpus = len(amd_gpu_names)

if amd_num_gpus > 0:
    amd_gpus = {x: amd_gpu_names.count(x) for x in amd_gpu_names}
    amd_gpu_arch = {}
    # get AMD gfx from GPU name
    for name in amd_gpus:
        found_name = False
        for key in amd_arch_mapping:
            name_cleaned = name.replace("AMD", "").strip()
            name_cleaned = name_cleaned.replace("(TM) ", "").strip()
            if re.search(key, name_cleaned, re.IGNORECASE):
                amd_gpu_arch[name] = amd_arch_mapping[key]
                found_name = True
                break

        if not found_name:
            raise RuntimeError("Unrecognized GPU name '{}'".format(name))

    cond_print("Found {} AMD GPU(s):".format(amd_num_gpus))
    for name in amd_gpus:
        cond_print("  {}x {}: {}".format(amd_gpus[name], name, amd_gpu_arch[name]))
    cond_print()

    plssvm_target_platforms += ";amd:" + ",".join({str(sm) for sm in amd_gpu_arch.values()})

# Intel GPU information
intel_gpu_names = []
for device in pylspci.parsers.SimpleParser().run():
    if re.search("VGA", str(device.cls), re.IGNORECASE):
        for key in intel_arch_mapping:
            if any(re.search(arch, str(device.device), re.IGNORECASE) for arch in intel_arch_mapping[key]):
                intel_gpu_names.append(str(device.device))
                break
intel_num_gpus = len(intel_gpu_names)

if intel_num_gpus > 0:
    intel_gpus = {x: intel_gpu_names.count(x) for x in intel_gpu_names}
    intel_gpu_arch = {}
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

    plssvm_target_platforms += ";intel:" + ",".join({str(sm) for sm in intel_gpu_arch.values()})

cond_print("Possible -DPLSSVM_TARGET_PLATFORMS entries:")
print("{}".format(plssvm_target_platforms))
